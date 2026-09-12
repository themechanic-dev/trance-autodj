"""Playlists: rows in the database, an .m3u file on disk for Liquidsoap.

Liquidsoap watches exactly one file, ``active.m3u``. Making a playlist active
means rewriting that file; Liquidsoap notices and picks it up without the
broadcast stopping. Nothing else is involved, which is why a hot reload cannot
half-fail.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import Session, select

from app.core.logging import get_logger
from app.core.paths import Paths
from app.models.entities import Playlist, PlaylistItem, PlaylistMode, Track, utcnow
from app.services.audio import analysis

log = get_logger(__name__)


@dataclass(frozen=True)
class PlaylistSummary:
    id: int
    name: str
    mode: str
    is_active: bool
    crossfade_s: float | None
    track_count: int
    duration_s: float
    missing_tracks: int
    rotation_position: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "is_active": self.is_active,
            "crossfade_s": self.crossfade_s,
            "track_count": self.track_count,
            "duration_s": round(self.duration_s, 1),
            "missing_tracks": self.missing_tracks,
            "rotation_position": self.rotation_position,
        }


def summarise(session: Session, playlist: Playlist) -> PlaylistSummary:
    items = session.exec(select(PlaylistItem).where(PlaylistItem.playlist_id == playlist.id)).all()
    tracks = [session.get(Track, item.track_id) for item in items]
    present = [t for t in tracks if t is not None]
    return PlaylistSummary(
        id=playlist.id or 0,
        name=playlist.name,
        mode=playlist.mode.value,
        is_active=playlist.is_active,
        crossfade_s=playlist.crossfade_s,
        track_count=len(present),
        duration_s=sum(t.duration_s for t in present if not t.missing),
        missing_tracks=sum(1 for t in present if t.missing),
        rotation_position=playlist.rotation_position,
    )


def create(session: Session, name: str, *, mode: PlaylistMode = PlaylistMode.SHUFFLE) -> Playlist:
    playlist = Playlist(name=name, mode=mode)
    session.add(playlist)
    session.flush()
    return playlist


def set_tracks(session: Session, playlist: Playlist, track_ids: list[int]) -> int:
    """Replace the playlist's contents, preserving the given order."""
    for item in session.exec(
        select(PlaylistItem).where(PlaylistItem.playlist_id == playlist.id)
    ).all():
        session.delete(item)
    session.flush()

    # Whoever called us may already have loaded playlist.items — ensure_default
    # does, to decide whether the library actually changed — and that
    # collection still holds the rows just deleted. Left alone, the save at the
    # end of this function cascades into them and raises "Instance
    # '<PlaylistItem>' has been deleted" instead of saving anything. Forget the
    # collection and let it load again from what is really there.
    session.expire(playlist, ["items"])

    added = 0
    for position, track_id in enumerate(track_ids):
        if session.get(Track, track_id) is None:
            continue
        session.add(PlaylistItem(playlist_id=playlist.id, track_id=track_id, position=position))
        added += 1
    playlist.updated_at = utcnow()
    session.add(playlist)
    return added


def tracks_of(
    session: Session, playlist: Playlist, *, include_missing: bool = False
) -> list[Track]:
    items = session.exec(
        select(PlaylistItem)
        .where(PlaylistItem.playlist_id == playlist.id)
        .order_by(PlaylistItem.position)
    ).all()
    tracks: list[Track] = []
    for item in items:
        track = session.get(Track, item.track_id)
        if track is None:
            continue
        if track.missing and not include_missing:
            continue
        tracks.append(track)
    return tracks


def active(session: Session) -> Playlist | None:
    return session.exec(select(Playlist).where(Playlist.is_active)).first()


def _annotations(track: Track, cfg) -> dict[str, str]:
    """Per-track values passed to Liquidsoap through the `annotate:` protocol.

    This is how a measured tempo reaches the transition: Liquidsoap reads
    these as metadata on the request, our crossfade function looks at them,
    and `cross` uses `liq_cross_duration` to buffer the right amount. There is
    no other channel — an .m3u carries no arbitrary fields, and writing them
    into the audio files themselves would be vandalism.
    """
    if cfg is None:
        return {}
    values: dict[str, str] = {}
    crossfade = cfg.audio.crossfade

    if track.bpm:
        values["bpm"] = f"{track.bpm:.2f}"

    if crossfade.beat_aligned and track.bpm:
        bar = (60.0 / track.bpm) * analysis.BEATS_PER_BAR
        estimate = analysis.TrackAnalysis(
            bpm=track.bpm,
            beat_offset_s=track.beat_offset_s or 0.0,
            confidence=track.bpm_confidence,
            method=track.analysis_method,
        )
        duration = analysis.bars_near(
            crossfade.duration_s,
            estimate,
            minimum=crossfade.min_duration_s,
            maximum=crossfade.max_duration_s,
        )
        values["liq_cross_duration"] = f"{duration:.3f}"
        values["bars"] = f"{round(duration / bar)}"

        if (
            crossfade.beat_align_cue_in
            and track.beat_offset_s
            and 0 < track.beat_offset_s <= crossfade.max_cue_in_s
        ):
            values["liq_cue_in"] = f"{track.beat_offset_s:.3f}"

    return values


def _annotate(path: Path, values: dict[str, str]) -> str:
    if not values:
        return str(path)
    fields = ",".join(f'{key}="{value}"' for key, value in values.items())
    return f"annotate:{fields}:{path}"


def rotation(session: Session) -> list[Playlist]:
    """The playlists in the rotation, in the order the operator arranged."""
    return list(
        session.exec(
            select(Playlist)
            .where(Playlist.rotation_position != None)  # noqa: E711
            .order_by(Playlist.rotation_position)
        ).all()
    )


def set_rotation(session: Session, playlist_ids: list[int]) -> list[Playlist]:
    """Replace the rotation with exactly these playlists, in this order.

    Positions are rewritten from zero every time rather than patched, so a
    reorder cannot leave gaps or duplicates behind — the list the operator
    sees is the list that is stored.
    """
    for playlist in session.exec(
        select(Playlist).where(Playlist.rotation_position != None)  # noqa: E711
    ).all():
        playlist.rotation_position = None
        session.add(playlist)
    session.flush()

    ordered: list[Playlist] = []
    for position, playlist_id in enumerate(playlist_ids):
        playlist = session.get(Playlist, playlist_id)
        if playlist is None:
            continue
        playlist.rotation_position = position
        playlist.updated_at = utcnow()
        session.add(playlist)
        ordered.append(playlist)
    session.flush()
    return ordered


def _ordered_tracks(session: Session, playlist: Playlist, rng: random.Random) -> list[Track]:
    """The tracks of one playlist, in the order they should be written.

    In a rotation the file order *is* the play order, so a playlist set to
    shuffle has to be shuffled here — Liquidsoap is reading straight through
    and will not do it for us. Re-saving the rotation therefore deals a fresh
    hand, which is the only variety a fixed file can offer.
    """
    tracks = tracks_of(session, playlist)
    if playlist.mode is PlaylistMode.SHUFFLE or playlist.mode is PlaylistMode.RANDOM:
        rng.shuffle(tracks)
    return tracks


def _render(paths: Paths, tracks: list[Track], cfg, header: str) -> list[str]:
    lines = ["#EXTM3U", f"# {header} — generated by Trance AutoDJ"]
    for track in tracks:
        label = f"{track.artist} - {track.title}" if track.artist else track.title
        lines.append(f"#EXTINF:{int(track.duration_s)},{label}")
        lines.append(_annotate(paths.music / track.relpath, _annotations(track, cfg)))
    return lines


def _write_lines(paths: Paths, lines: list[str]) -> None:
    paths.active_playlist.parent.mkdir(parents=True, exist_ok=True)
    # Written atomically: Liquidsoap watches this file and must never read a
    # half-written one.
    temporary = paths.active_playlist.with_suffix(".m3u.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(paths.active_playlist)


def write_active(session: Session, paths: Paths, cfg=None) -> int:
    """Write whatever the station should be playing, and say how many tracks.

    One entry point on purpose: every caller that changes the library, the
    playlists or the rotation ends here, so there is a single answer to "what
    is in the file Liquidsoap is reading".
    """
    playback = getattr(getattr(cfg, "audio", None), "playlist_playback", "single")
    if playback != "rotation":
        playlist = active(session)
        if playlist is None:
            return 0
        return write_m3u(session, playlist, paths, cfg)

    lists = rotation(session)
    if not lists:
        # Rotation asked for, nothing in it. Falling back to the active list
        # keeps the station on air instead of handing Liquidsoap an empty file
        # and dropping it onto the silent safety source.
        playlist = active(session)
        if playlist is None:
            return 0
        log.warning("rotation is empty; playing the active playlist instead")
        return write_m3u(session, playlist, paths, cfg)

    rng = random.Random()
    tracks: list[Track] = []
    for playlist in lists:
        tracks.extend(_ordered_tracks(session, playlist, rng))

    header = "rotation: " + " → ".join(p.name for p in lists)
    _write_lines(paths, _render(paths, tracks, cfg, header))
    log.info(
        "active playlist written",
        extra={
            "playback": "rotation",
            "playlists": [p.name for p in lists],
            "tracks": len(tracks),
            "path": str(paths.active_playlist),
        },
    )
    return len(tracks)


def write_m3u(session: Session, playlist: Playlist, paths: Paths, cfg=None) -> int:
    """Write one playlist to active.m3u. Returns how many tracks it holds.

    Extended M3U so Liquidsoap has durations and titles even for files whose
    tags it would otherwise have to open and read, plus `annotate:` fields
    carrying anything we have measured about the track.
    """
    tracks = tracks_of(session, playlist)
    lines = _render(paths, tracks, cfg, playlist.name)

    paths.active_playlist.parent.mkdir(parents=True, exist_ok=True)
    # Written atomically: Liquidsoap watches this file and must never read a
    # half-written one.
    temporary = paths.active_playlist.with_suffix(".m3u.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(paths.active_playlist)

    log.info(
        "active playlist written",
        extra={
            "playlist": playlist.name,
            "tracks": len(tracks),
            "path": str(paths.active_playlist),
        },
    )
    return len(tracks)


def activate(session: Session, playlist: Playlist, paths: Paths, cfg=None) -> int:
    """Make this the playing playlist and write the file Liquidsoap watches."""
    for other in session.exec(select(Playlist).where(Playlist.is_active)).all():
        if other.id != playlist.id:
            other.is_active = False
            session.add(other)
    playlist.is_active = True
    playlist.updated_at = utcnow()
    session.add(playlist)
    session.flush()
    # Through write_active, not write_m3u: in a rotation, activating a list
    # decides what plays when the operator switches back to a single list, and
    # must not quietly replace the rotation that is playing now.
    return write_active(session, paths, cfg)


#: The auto-managed playlist. Its contents track the library; anything the
#: operator builds by hand is left alone.
DEFAULT_PLAYLIST_NAME = "All tracks"


def ensure_default(session: Session, paths: Paths, cfg=None) -> Playlist:
    """Guarantee there is an active playlist, and keep the automatic one fresh.

    Returning early when *any* playlist was already active looked right and
    was wrong: on a new install the empty "All tracks" is created at startup,
    before any music exists, and then never refilled — so the first scan
    imported four tracks into a playlist that stayed empty and the station
    played silence.

    So: a hand-made active playlist is never touched, but the automatic one is
    resynchronised with the library every time.
    """
    existing = active(session)
    if existing is not None and existing.name != DEFAULT_PLAYLIST_NAME:
        return existing

    playlist = session.exec(select(Playlist).where(Playlist.name == DEFAULT_PLAYLIST_NAME)).first()
    if playlist is None:
        playlist = create(session, DEFAULT_PLAYLIST_NAME)

    track_ids = [
        t.id
        for t in session.exec(
            select(Track).where(Track.missing == False).order_by(Track.relpath)  # noqa: E712
        ).all()
        if t.id is not None
    ]
    current = [item.track_id for item in playlist.items] if playlist.id else []
    if current != track_ids or not playlist.is_active:
        set_tracks(session, playlist, track_ids)
        activate(session, playlist, paths, cfg)
    return playlist


def refresh_active(session: Session, paths: Paths, cfg=None) -> int:
    """Rewrite the file Liquidsoap reads for the current settings.

    Called when something that is not the library itself changes what should
    be playing: the rotation order, or the switch between one list and many.
    """
    return write_active(session, paths, cfg)
