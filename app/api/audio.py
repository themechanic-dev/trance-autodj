"""Audio engine endpoints: status, control, library, playlists."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_runtime
from app.core.db import get_session, session_scope
from app.core.logging import get_logger
from app.core.runtime import Runtime
from app.models.entities import (
    GenerationJob,
    JobKind,
    JobStatus,
    Playlist,
    PlaylistMode,
    PlayLogEntry,
    Track,
)
from app.services.audio import library, nowplaying, playlists, playlog
from app.services.audio.manager import AudioManager

log = get_logger(__name__)

router = APIRouter(prefix="/api/audio", tags=["audio"])

# Reject an upload that is obviously not music before writing it to disk.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

#: Not a limit on ambition — a hundred tracks on a slow NAS is a week of
#: composing, and that is the operator's call. This only stops a typo from
#: creating a million rows. Ten thousand tracks is a month of music.
MAX_GENERATED_AT_ONCE = 10_000


def get_audio(request: Request) -> AudioManager:
    manager = getattr(request.app.state, "audio", None)
    if manager is None:
        raise HTTPException(503, "the audio engine is not available in this process")
    return manager


# --------------------------------------------------------------------------
# status and control
# --------------------------------------------------------------------------


@router.get("/status")
def status(audio: Annotated[AudioManager, Depends(get_audio)]) -> dict[str, Any]:
    return audio.status().as_dict()


@router.get("/nowplaying")
def now_playing(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    return nowplaying.read(runtime.paths.nowplaying_file).as_dict()


@router.post("/start")
def start(audio: Annotated[AudioManager, Depends(get_audio)]) -> dict[str, Any]:
    audio.start()
    return audio.status().as_dict()


@router.post("/stop")
def stop(audio: Annotated[AudioManager, Depends(get_audio)]) -> dict[str, Any]:
    audio.stop()
    return audio.status().as_dict()


@router.post("/restart")
def restart(audio: Annotated[AudioManager, Depends(get_audio)]) -> dict[str, Any]:
    audio.restart()
    return audio.status().as_dict()


@router.post("/skip")
def skip(audio: Annotated[AudioManager, Depends(get_audio)]) -> dict[str, Any]:
    return {"skipped": audio.skip_track()}


@router.get("/logs/{which}")
def logs(
    which: str,
    audio: Annotated[AudioManager, Depends(get_audio)],
    count: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    if which not in {"icecast", "liquidsoap"}:
        raise HTTPException(404, "no such process")
    return {"process": which, "lines": audio.logs(which, count)}


@router.get("/config/liquidsoap")
def liquidsoap_config(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    """The generated script, for when you need to see what is actually running.

    The Icecast config is deliberately not exposed: it contains the source
    password in plain text.
    """
    path = runtime.paths.liquidsoap_script
    if not path.is_file():
        raise HTTPException(404, "no script has been generated yet")
    return {"path": str(path), "content": path.read_text(encoding="utf-8")}


# --------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------


def _track_json(track: Track) -> dict[str, Any]:
    return {
        "id": track.id,
        "relpath": track.relpath,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "genre": track.genre,
        "year": track.year,
        "duration_s": round(track.duration_s, 1),
        "bitrate_k": track.bitrate_k,
        "size_mb": round(track.size_bytes / 1024**2, 1),
        "bpm": track.bpm,
        "musical_key": track.musical_key,
        "play_count": track.play_count,
        "missing": track.missing,
        "flagged": track.flagged,
        "flag_note": track.flag_note,
        "added_at": track.added_at.isoformat(),
    }


@router.get("/tracks")
def list_tracks(
    session: Annotated[Session, Depends(get_session)],
    search: str | None = None,
    include_missing: bool = True,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    statement = select(Track)
    if not include_missing:
        statement = statement.where(Track.missing == False)  # noqa: E712
    if search:
        pattern = f"%{search.lower()}%"
        statement = statement.where(
            Track.title.ilike(pattern)  # type: ignore[union-attr]
            | Track.artist.ilike(pattern)  # type: ignore[union-attr]
            | Track.album.ilike(pattern)  # type: ignore[union-attr]
            | Track.relpath.ilike(pattern)  # type: ignore[union-attr]
        )
    total = len(session.exec(statement).all())
    tracks = session.exec(
        statement.order_by(Track.artist, Track.title).offset(offset).limit(limit)
    ).all()
    return {
        "tracks": [_track_json(t) for t in tracks],
        "total": total,
        "duration_s": round(library.total_duration(session), 1),
    }


@router.post("/tracks/scan")
def scan_library(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    with session_scope() as session:
        result = library.scan(session, runtime.paths)
        playlists.ensure_default(session, runtime.paths, runtime.config)
    return result.as_dict()


@router.post("/tracks/analyse")
def analyse_tracks(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    limit: int = Query(default=25, ge=1, le=500),
    force: bool = False,
) -> dict[str, Any]:
    """Measure the tempo of tracks that have not been measured yet.

    Bounded, so it can be pressed repeatedly and make progress each time
    instead of blocking on a large library.
    """
    with session_scope() as session:
        result = library.analyse_pending(
            session, runtime.paths, runtime.config, limit=limit, force=force
        )
        # A measured tempo changes what the playlist says, so rewrite it.
        active = playlists.active(session)
        if active is not None and result["analysed"]:
            playlists.write_m3u(session, active, runtime.paths, runtime.config)
    return result


@router.get("/tracks/analysis")
def analysis_status(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    summary = library.analysis_summary(session)
    crossfade = runtime.config.audio.crossfade
    return {
        **summary,
        "enabled": runtime.config.audio.analysis.enabled,
        "beat_aligned": crossfade.beat_aligned,
        "beat_align_cue_in": crossfade.beat_align_cue_in,
        "bpm_range": [runtime.config.audio.analysis.min_bpm, runtime.config.audio.analysis.max_bpm],
    }


@router.get("/tracks/duplicates")
def duplicates(session: Annotated[Session, Depends(get_session)]) -> dict[str, Any]:
    groups = library.find_duplicates(session)
    return {
        "groups": [
            {"hash": digest[:16], "tracks": [_track_json(t) for t in tracks]}
            for digest, tracks in groups.items()
        ],
        "count": len(groups),
    }


@router.get("/upload/limits")
def upload_limits() -> dict[str, Any]:
    """What the uploader may send, so the page can refuse a file up front.

    Finding out that a 700 MB file was the wrong type after uploading it is a
    poor way to learn the rule.
    """
    return {
        "max_bytes": MAX_UPLOAD_BYTES,
        "suffixes": sorted(library.SUPPORTED_SUFFIXES),
    }


@router.post("/tracks/upload")
async def upload_tracks(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    files: list[UploadFile] = File(...),
    playlist: str = Form(""),
) -> dict[str, Any]:
    """Import files, and optionally put them in a playlist of that name.

    `playlist` is what turns "upload a folder" into something useful: the page
    sends the folder's name with its music, and the tracks land in a list
    called that instead of in one undifferentiated library the operator then
    has to sort by hand.
    """
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    original_name: dict[str, str] = {}
    runtime.paths.music.mkdir(parents=True, exist_ok=True)

    for upload in files:
        name = Path(upload.filename or "").name
        if not name:
            rejected.append({"file": "?", "reason": "no filename"})
            continue
        if Path(name).suffix.lower() not in library.SUPPORTED_SUFFIXES:
            rejected.append({"file": name, "reason": "unsupported file type"})
            continue

        target = runtime.paths.music / name
        # Never overwrite: two different tracks can share a filename.
        counter = 1
        while target.exists():
            target = runtime.paths.music / f"{Path(name).stem} ({counter}){Path(name).suffix}"
            counter += 1

        written = 0
        try:
            with target.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise ValueError("file is too large")
                    handle.write(chunk)
        except (OSError, ValueError) as exc:
            target.unlink(missing_ok=True)
            rejected.append({"file": name, "reason": str(exc)})
            continue
        accepted.append(target.name)
        # The name on disk can differ from the name uploaded, because two
        # different tracks are allowed to share a filename. Report the one the
        # operator recognises.
        original_name[target.name] = name

    duplicates: list[str] = []
    with session_scope() as session:
        # A folder uploaded twice must not double the library. The scan would
        # happily import the second copy under "name (1)", and the operator
        # would have no way of telling which is which.
        for name in list(accepted):
            path = runtime.paths.music / name
            if not path.is_file():
                continue
            digest = library.content_hash(path)
            twin = session.exec(
                select(Track).where(Track.sha256 == digest, Track.relpath != name)
            ).first()
            if twin is not None and (runtime.paths.music / twin.relpath).is_file():
                path.unlink(missing_ok=True)
                accepted.remove(name)
                accepted.append(twin.relpath)
                duplicates.append(original_name.get(name, name))

        library.scan(session, runtime.paths)
        session.flush()

        added_to = None
        if playlist.strip():
            added_to = _put_in_playlist(session, runtime, playlist.strip(), accepted)

        playlists.ensure_default(session, runtime.paths, runtime.config)

    log.info(
        "upload finished",
        extra={
            "accepted": len(accepted),
            "rejected": len(rejected),
            "duplicates": len(duplicates),
            "playlist": playlist or None,
        },
    )
    return {
        "accepted": accepted,
        "rejected": rejected,
        "duplicates": duplicates,
        "playlist": added_to,
    }


def _put_in_playlist(session, runtime, name: str, relpaths: list[str]) -> dict[str, Any]:
    """Add these files to the named playlist, creating it if it is new.

    Appending rather than replacing: uploading the rest of an album later
    should top the list up, not empty it.
    """
    target = session.exec(select(Playlist).where(Playlist.name == name)).first()
    if target is None:
        target = playlists.create(session, name)
        session.flush()

    existing = [item.track_id for item in target.items]
    for relpath in relpaths:
        track = session.exec(select(Track).where(Track.relpath == relpath)).first()
        if track is not None and track.id not in existing:
            existing.append(track.id)

    playlists.set_tracks(session, target, existing)
    session.flush()
    if target.is_active or target.rotation_position is not None:
        playlists.refresh_active(session, runtime.paths, runtime.config)
    return playlists.summarise(session, target).as_dict()


class GenerateIn(BaseModel):
    """What to compose. Every field has a sane default so the button works."""

    count: int = Field(default=3, ge=1, le=MAX_GENERATED_AT_ONCE)
    minutes: float = Field(default=5.0, ge=1.0, le=12.0)
    playlist: str = ""
    #: "mixed" lets the seed choose, so a batch is not one sub-genre; a name
    #: pins every track in the batch to that style.
    style: str = "mixed"
    #: Given, the first track uses it and the rest follow on. Left out, the
    #: clock picks — so pressing the button twice never makes the same music,
    #: while a track worth keeping can always be rebuilt from its number.
    seed: int | None = Field(default=None, ge=0, le=99_999_999)


@router.post("/tracks/generate")
def generate_tracks(
    session: Annotated[Session, Depends(get_session)],
    payload: GenerateIn,
) -> dict[str, Any]:
    """Compose music from nothing and put it in the library.

    Nothing is composed here: the tracks go into a queue in the database and
    a worker takes them one at a time, in a process niced below the
    broadcast. A batch of a hundred is a day on a desktop and a week on a
    NAS, and it survives a restart either way — so this returns at once.
    """
    from app.services.music import queue, styles

    if payload.style != styles.MIXED and payload.style not in styles.STYLES:
        raise HTTPException(
            422, f"no such style: {payload.style!r}; one of {', '.join(styles.STYLES)} or mixed"
        )

    first = payload.seed if payload.seed is not None else int(time.time()) % 100_000_000
    seeds = [(first + index * 7919) % 100_000_000 for index in range(payload.count)]
    queue.enqueue(
        session,
        seeds=seeds,
        minutes=payload.minutes,
        style=payload.style,
        playlist=payload.playlist.strip(),
    )
    log.info("music generation queued", extra={"count": len(seeds), "style": payload.style})
    return {"queued": len(seeds), "seeds": seeds}


@router.delete("/tracks/generate/queue")
def cancel_generation(session: Annotated[Session, Depends(get_session)]) -> dict[str, Any]:
    """Change of mind. The track being composed finishes; the rest never start."""
    from app.services.music import queue

    cancelled = queue.cancel_waiting(session)
    log.info("music generation cancelled", extra={"cancelled": cancelled})
    return {"cancelled": cancelled}


@router.get("/tracks/generate/styles")
def generation_styles() -> dict[str, Any]:
    from app.services.music import styles

    return {
        "mixed": styles.MIXED,
        "styles": [
            {"name": s.name, "blurb": s.blurb, "bpm": list(s.bpm_range)}
            for s in styles.STYLES.values()
        ],
    }


@router.get("/tracks/generate/status")
def generation_status(session: Annotated[Session, Depends(get_session)]) -> dict[str, Any]:
    from app.services.music import queue

    jobs = session.exec(
        select(GenerationJob)
        .where(GenerationJob.kind == JobKind.TRACK)
        .where(GenerationJob.status != JobStatus.QUEUED)
        .order_by(GenerationJob.created_at.desc())  # type: ignore[union-attr]
        .limit(12)
    ).all()
    return {
        "running": sum(1 for job in jobs if job.status is JobStatus.RUNNING),
        "waiting": queue.waiting_count(session),
        "jobs": [
            {
                "id": job.id,
                "status": job.status.value,
                "step": job.step,
                "seed": job.result_id or json.loads(job.params_json or "{}").get("seed", ""),
                "error": job.error,
            }
            for job in jobs
        ],
    }


@router.post("/tracks/purge-missing")
def purge_missing(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    """Forget every track whose file is gone.

    Missing rows are kept on purpose — a mount that comes back should not cost
    you your playlists — but once you have decided a file is gone for good,
    there has to be a way to say so without editing the database by hand.
    """
    gone = session.exec(select(Track).where(Track.missing == True)).all()  # noqa: E712
    for track in gone:
        session.delete(track)
    session.flush()
    playlists.ensure_default(session, runtime.paths, runtime.config)
    log.info("purged missing tracks", extra={"count": len(gone)})
    return {"purged": len(gone)}


def _erase(runtime: Runtime, session: Session, tracks: list[Track]) -> tuple[int, list[str]]:
    """Remove tracks from the library and their files from the disk.

    Rows first, files second, and a flush in between: a failure at the request
    teardown happens after the response has gone out, so a caller told that
    thirty tracks were removed must not be left with thirty deleted files and
    thirty surviving rows. Losing the file and keeping the row is the bad
    direction; the other way round is a rescan away from correct.
    """
    names = []
    for track in tracks:
        names.append(track.relpath)
        session.delete(track)
    session.flush()

    for relpath in names:
        (runtime.paths.music / relpath).unlink(missing_ok=True)

    playlists.ensure_default(session, runtime.paths, runtime.config)
    playlists.refresh_active(session, runtime.paths, runtime.config)
    return len(names), names


class EraseLibraryIn(BaseModel):
    """How many tracks the caller believes it is about to destroy.

    The dashboard sends back the number it just showed. If an upload or a scan
    landed in between, the counts differ and nothing is deleted — the operator
    is asked again with the real number rather than silently taking more than
    they agreed to.
    """

    confirm_count: int = Field(..., ge=0)


@router.delete("/tracks")
def erase_library(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
    payload: EraseLibraryIn,
) -> dict[str, Any]:
    """Empty the music library: every track, every file.

    Deleting eighty-nine tracks one button at a time is not a workflow, and
    changing the whole repertoire at once is a normal thing to want — a
    copyright claim, a change of style, a station handed to someone else.
    """
    tracks = list(session.exec(select(Track)).all())
    if len(tracks) != payload.confirm_count:
        raise HTTPException(
            409,
            f"the library holds {len(tracks)} tracks, not {payload.confirm_count}; "
            "it changed since you looked — check it and try again",
        )

    count, _ = _erase(runtime, session, tracks)
    log.info("library erased", extra={"count": count})
    return {"deleted": count}


def _playlog_json(entry: PlayLogEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "track_id": entry.track_id,
        "relpath": entry.relpath,
        "title": entry.title,
        "artist": entry.artist,
        # Marked as UTC on the way out: the column holds no offset, and a
        # bare timestamp is read by every browser as local time.
        "started_at": playlog.as_utc(entry.started_at).isoformat(),
        "ended_at": (playlog.as_utc(entry.ended_at).isoformat() if entry.ended_at else None),
    }


@router.get("/playlog")
def read_playlog(
    session: Annotated[Session, Depends(get_session)],
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    return {"entries": [_playlog_json(e) for e in playlog.recent(session, limit)]}


@router.get("/playlog/at")
def playlog_at(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
    when: str = Query(..., description="a time, read in the station's own clock"),
) -> dict[str, Any]:
    """What was on air then — the question a copyright notice actually asks.

    The time is taken as station time unless it says otherwise, because a
    claim is read off a screen in local time and asking the operator to
    convert it is asking at the worst possible moment.
    """
    try:
        moment = playlog.parse_moment(when, runtime.config.app.timezone)
    except ValueError as exc:
        raise HTTPException(422, f"that is not a time I can read: {exc}") from exc

    entry = playlog.at(session, moment)
    return {
        "when": moment.isoformat(),
        "entry": _playlog_json(entry) if entry else None,
    }


class FlagIn(BaseModel):
    note: str = Field(default="", max_length=500)


@router.post("/tracks/flagged")
def flag_track(
    session: Annotated[Session, Depends(get_session)],
    payload: FlagIn,
    track_id: int = Query(..., ge=1),
) -> dict[str, Any]:
    """Mark a track without removing it.

    A claim lands while the station is on air, and taking the file out from
    under Liquidsoap mid-broadcast is how you get silence. Flag now, sweep
    later.
    """
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(404, "no such track")
    track.flagged = True
    track.flag_note = payload.note.strip()
    log.info("track flagged", extra={"track": track.relpath, "note": track.flag_note})
    return _track_json(track)


@router.delete("/tracks/flagged")
def clear_flags(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
    remove: bool = Query(default=False, description="delete the files as well"),
) -> dict[str, Any]:
    """Unflag everything, or delete the lot — the sweep after the triage."""
    flagged = list(session.exec(select(Track).where(Track.flagged == True)).all())  # noqa: E712
    if not remove:
        for track in flagged:
            track.flagged = False
            track.flag_note = ""
        return {"cleared": len(flagged), "deleted": 0}

    deleted, _ = _erase(runtime, session, flagged)
    log.info("flagged tracks removed", extra={"count": deleted})
    return {"cleared": 0, "deleted": deleted}


@router.delete("/tracks/{track_id}")
def delete_track(
    track_id: int,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
    delete_file: bool = True,
) -> dict[str, Any]:
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(404, "no such track")

    relpath = track.relpath
    session.delete(track)
    # Flush here rather than letting the request's teardown commit. A failure
    # at teardown happens after the response has been sent, so the caller is
    # told the track was deleted when it was not — and the file below would
    # already be gone. Deleting from the database first, and only then from
    # the disk, means a failure loses nothing.
    session.flush()

    if delete_file:
        (runtime.paths.music / relpath).unlink(missing_ok=True)

    # The automatic playlist and the m3u Liquidsoap reads have to stop
    # mentioning it too.
    playlists.ensure_default(session, runtime.paths, runtime.config)
    log.info("track deleted", extra={"track": relpath, "file_removed": delete_file})
    return {"deleted": track_id, "relpath": relpath}


@router.get("/tracks/{track_id}/file")
def track_file(
    track_id: int,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> FileResponse:
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(404, "no such track")
    path = runtime.paths.music / track.relpath
    if not path.is_file():
        raise HTTPException(404, "the file is gone")
    return FileResponse(path, filename=track.filename)


# --------------------------------------------------------------------------
# playlists
# --------------------------------------------------------------------------


@router.get("/playlists")
def list_playlists(session: Annotated[Session, Depends(get_session)]) -> dict[str, Any]:
    rows = session.exec(select(Playlist).order_by(Playlist.name)).all()
    return {"playlists": [playlists.summarise(session, p).as_dict() for p in rows]}


@router.post("/playlists")
def create_playlist(
    name: str,
    session: Annotated[Session, Depends(get_session)],
    mode: PlaylistMode = PlaylistMode.SHUFFLE,
) -> dict[str, Any]:
    if session.exec(select(Playlist).where(Playlist.name == name)).first():
        raise HTTPException(409, "a playlist with that name already exists")
    playlist = playlists.create(session, name, mode=mode)
    return playlists.summarise(session, playlist).as_dict()


class RotationIn(BaseModel):
    """The rotation, top to bottom, as playlist ids."""

    playlist_ids: list[int]


@router.post("/playlists/prune-empty")
def prune_empty_playlists(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    """Delete every hand-made playlist that has no tracks left in it.

    Emptying the library leaves the lists behind, holding nothing and still
    sitting in the rotation — which is the same one-at-a-time tidying the bulk
    deletes were added to stop. The automatic "All tracks" stays: it mirrors
    the library, so being empty is the correct state for it rather than a
    leftover, and the station needs something active to fall back to.

    The active list is not spared if it is empty: there is nothing in it to
    play, and ensure_default puts the automatic one back on air afterwards.
    """
    doomed = [
        playlist
        for playlist in session.exec(select(Playlist)).all()
        if playlist.name != playlists.DEFAULT_PLAYLIST_NAME and not playlist.items
    ]
    names = [playlist.name for playlist in doomed]
    for playlist in doomed:
        session.delete(playlist)
    session.flush()

    # Deleting the active list leaves nothing on air, and deleting a list that
    # was in the rotation changes what plays next; both are put right here.
    playlists.ensure_default(session, runtime.paths, runtime.config)
    playlists.refresh_active(session, runtime.paths, runtime.config)
    log.info("pruned empty playlists", extra={"count": len(names), "names": names})
    return {"removed": len(names), "names": names}


@router.get("/playlists/rotation")
def get_rotation(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    lists = playlists.rotation(session)
    return {
        "playback": runtime.config.audio.playlist_playback,
        "playlist_ids": [p.id for p in lists],
        "playlists": [playlists.summarise(session, p).as_dict() for p in lists],
    }


@router.put("/playlists/rotation")
def put_rotation(
    payload: RotationIn,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    """Replace the rotation, in this order, and rewrite what is playing."""
    lists = playlists.set_rotation(session, payload.playlist_ids)
    session.flush()
    tracks = playlists.refresh_active(session, runtime.paths, runtime.config)
    log.info("rotation set", extra={"playlists": [p.name for p in lists], "tracks": tracks})
    return {
        "playlists": [playlists.summarise(session, p).as_dict() for p in lists],
        "tracks": tracks,
    }


@router.get("/playlists/{playlist_id}")
def playlist_detail(
    playlist_id: int, session: Annotated[Session, Depends(get_session)]
) -> dict[str, Any]:
    playlist = session.get(Playlist, playlist_id)
    if playlist is None:
        raise HTTPException(404, "no such playlist")
    return {
        **playlists.summarise(session, playlist).as_dict(),
        "tracks": [
            _track_json(t) for t in playlists.tracks_of(session, playlist, include_missing=True)
        ],
    }


@router.put("/playlists/{playlist_id}/tracks")
def set_playlist_tracks(
    playlist_id: int,
    track_ids: list[int],
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    playlist = session.get(Playlist, playlist_id)
    if playlist is None:
        raise HTTPException(404, "no such playlist")
    added = playlists.set_tracks(session, playlist, track_ids)
    # Editing a list that is on air — either as the active one or as part of
    # the rotation — has to reach the file Liquidsoap is reading.
    if playlist.is_active or playlist.rotation_position is not None:
        session.flush()
        playlists.refresh_active(session, runtime.paths, runtime.config)
    return {"playlist": playlist.name, "tracks": added}


@router.post("/playlists/{playlist_id}/activate")
def activate_playlist(
    playlist_id: int,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
    request: Request,
) -> dict[str, Any]:
    playlist = session.get(Playlist, playlist_id)
    if playlist is None:
        raise HTTPException(404, "no such playlist")
    count = playlists.activate(session, playlist, runtime.paths, runtime.config)

    # Liquidsoap watches the file, but nudge it as well: inotify does not
    # fire on every filesystem a container might be given.
    manager = getattr(request.app.state, "audio", None)
    if manager is not None:
        manager.reload_playlist()
    return {"activated": playlist.name, "tracks": count}


@router.delete("/playlists/{playlist_id}")
def delete_playlist(
    playlist_id: int,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
    with_tracks: bool = Query(default=False),
) -> dict[str, Any]:
    """Delete a playlist, and with `with_tracks` the music that only it holds.

    Only what only it holds: a track that another hand-made playlist also
    lists stays, because deleting the file would quietly empty that other list
    too. The automatic "All tracks" does not count as another list — it mirrors
    the library, so counting it would make nothing ever exclusive.
    """
    playlist = session.get(Playlist, playlist_id)
    if playlist is None:
        raise HTTPException(404, "no such playlist")
    if playlist.is_active:
        raise HTTPException(409, "cannot delete the active playlist")

    doomed: list[Track] = []
    kept = 0
    if with_tracks:
        for track in [item.track for item in playlist.items if item.track is not None]:
            elsewhere = [
                item.playlist
                for item in track.items
                if item.playlist is not None
                and item.playlist.id != playlist_id
                and item.playlist.name != playlists.DEFAULT_PLAYLIST_NAME
            ]
            if elsewhere:
                kept += 1
            else:
                doomed.append(track)

    session.delete(playlist)
    session.flush()
    # The rows joining these tracks to the playlist are gone now, but each
    # Track still holds the deleted PlaylistItem objects in its loaded
    # collection, so the cascade below would try to delete them a second time
    # and SQLAlchemy would warn that the rows it expected were not there.
    for track in doomed:
        session.expire(track, ["items"])

    deleted = 0
    if doomed:
        deleted, _ = _erase(runtime, session, doomed)
    log.info(
        "playlist deleted",
        extra={"playlist": playlist_id, "tracks_deleted": deleted, "tracks_kept": kept},
    )
    return {"deleted": playlist_id, "tracks_deleted": deleted, "tracks_kept": kept}


@router.get("/disk")
def disk_usage(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    usage = shutil.disk_usage(runtime.paths.root)
    return {
        "total_gb": round(usage.total / 1024**3, 1),
        "free_gb": round(usage.free / 1024**3, 1),
        "used_pct": round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
    }
