"""The music library: scanning files, reading tags, spotting duplicates.

The filesystem is the source of truth here as well. A row describes a file; a
file with no row gets one on the next scan, and a row whose file has gone is
marked missing rather than deleted, so an unmounted disk does not silently
empty every playlist.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import Session, select

from app.core.logging import get_logger
from app.core.paths import Paths
from app.models.entities import Track, utcnow
from app.services.audio import analysis

log = get_logger(__name__)

# Everything ffmpeg decodes and mutagen can name. `.wma` is here because it
# works, not because it is fashionable: ffmpeg carries wmav1, wmav2, wmapro
# and wmalossless, and Liquidsoap decodes through ffmpeg.
SUPPORTED_SUFFIXES = frozenset({".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma"})

# Hashing every byte of a 60 MB FLAC to spot duplicates is wasteful. The first
# and last chunk plus the size is enough to catch re-uploads of the same file
# while staying fast on a large library.
HASH_CHUNK = 1024 * 1024


@dataclass
class ScanResult:
    added: int = 0
    updated: int = 0
    missing: int = 0
    restored: int = 0
    skipped: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def as_dict(self) -> dict[str, object]:
        return {
            "added": self.added,
            "updated": self.updated,
            "missing": self.missing,
            "restored": self.restored,
            "skipped": self.skipped,
            "errors": self.errors[:20],
        }


def content_hash(path: Path) -> str:
    """A cheap content fingerprint: size plus the head and tail of the file."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(HASH_CHUNK))
        if size > HASH_CHUNK * 2:
            handle.seek(-HASH_CHUNK, 2)
            digest.update(handle.read(HASH_CHUNK))
    return digest.hexdigest()


@dataclass
class TrackTags:
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: int | None = None
    duration_s: float = 0.0
    bitrate_k: int = 0
    sample_rate: int = 0
    has_artwork: bool = False


def _first(value: object) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def read_tags(path: Path) -> TrackTags:
    """Read what tags there are. A file with none is still a usable track."""
    try:
        import mutagen
    except ImportError:  # pragma: no cover - mutagen is a hard requirement
        return TrackTags()

    try:
        audio = mutagen.File(path, easy=True)
    except Exception as exc:
        log.warning("could not read tags from %s: %s", path.name, exc)
        # Still name it. A file with unreadable tags is a playable file with
        # an ugly name, not a nameless row in the library.
        return TrackTags(title=path.stem)

    if audio is None:
        return TrackTags(title=path.stem)

    tags = TrackTags()
    data = dict(audio.tags or {})
    # ASF (.wma) keeps its own tag names and mutagen's "easy" interface does
    # not normalise them: the title arrives only because ASF happens to carry
    # a lowercase "title" as well, while the artist is "Author" and the rest
    # live under "WM/". Without the fallbacks a WMA library imports as a list
    # of titles by nobody.
    tags.title = _first(data.get("title") or data.get("Title"))
    tags.artist = _first(
        data.get("artist")
        or data.get("albumartist")
        or data.get("Author")
        or data.get("WM/AlbumArtist")
    )
    tags.album = _first(data.get("album") or data.get("WM/AlbumTitle"))
    tags.genre = _first(data.get("genre") or data.get("WM/Genre"))

    raw_year = _first(data.get("date") or data.get("year") or data.get("WM/Year"))
    if raw_year[:4].isdigit():
        tags.year = int(raw_year[:4])

    info = getattr(audio, "info", None)
    if info is not None:
        tags.duration_s = float(getattr(info, "length", 0.0) or 0.0)
        tags.bitrate_k = int(getattr(info, "bitrate", 0) or 0) // 1000
        tags.sample_rate = int(getattr(info, "sample_rate", 0) or 0)

    # Artwork lives on the non-easy interface.
    try:
        raw = mutagen.File(path)
        if raw is not None:
            tags.has_artwork = bool(
                getattr(raw, "pictures", None)
                or any(k.startswith("APIC") for k in (raw.tags or {}))
                or "covr" in (raw.tags or {})
            )
    except Exception as exc:
        log.debug("no artwork read from %s: %s", path.name, exc)

    if not tags.title:
        tags.title = path.stem
    return tags


def music_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def scan(session: Session, paths: Paths, *, rehash: bool = False) -> ScanResult:
    """Reconcile the music directory with the tracks table."""
    result = ScanResult()
    on_disk = music_files(paths.music)
    by_relpath = {str(p.relative_to(paths.music)): p for p in on_disk}

    rows = {t.relpath: t for t in session.exec(select(Track)).all()}

    for relpath, path in by_relpath.items():
        try:
            stat = path.stat()
        except OSError as exc:
            result.errors.append(f"{relpath}: {exc}")
            continue

        row = rows.get(relpath)
        unchanged = (
            row is not None
            and row.size_bytes == stat.st_size
            and not row.missing
            and row.analysed_at is not None
        )
        if unchanged and not rehash:
            result.skipped += 1
            continue

        try:
            tags = read_tags(path)
            digest = content_hash(path)
        except OSError as exc:
            result.errors.append(f"{relpath}: {exc}")
            continue

        if row is None:
            row = Track(relpath=relpath)
            result.added += 1
        elif row.missing:
            result.restored += 1
        else:
            result.updated += 1

        row.filename = path.name
        row.title = tags.title
        row.artist = tags.artist
        row.album = tags.album
        row.genre = tags.genre
        row.year = tags.year
        row.duration_s = tags.duration_s
        row.bitrate_k = tags.bitrate_k
        row.sample_rate = tags.sample_rate
        row.size_bytes = stat.st_size
        row.sha256 = digest
        row.missing = False
        row.analysed_at = utcnow()
        session.add(row)

    for relpath, row in rows.items():
        if relpath not in by_relpath and not row.missing:
            row.missing = True
            session.add(row)
            result.missing += 1

    log.info("library scanned", extra=result.as_dict())
    return result


def analyse_pending(
    session: Session,
    paths: Paths,
    cfg,
    *,
    limit: int = 25,
    force: bool = False,
) -> dict[str, int]:
    """Measure the tempo of tracks that do not have one yet.

    Kept out of :func:`scan` on purpose. Reading tags is milliseconds per file;
    measuring tempo decodes two minutes of audio and runs an FFT over it, so a
    first scan of a large library would otherwise block for a very long time
    behind something the station does not need in order to start playing.

    Bounded by ``limit`` so it can be called repeatedly — from the dashboard,
    or by the generator while the machine is idle — and make progress each
    time without ever running long.
    """
    if not cfg.audio.analysis.enabled and not force:
        return {"analysed": 0, "failed": 0, "remaining": 0, "skipped": True}

    statement = select(Track).where(Track.missing == False)  # noqa: E712
    if not force:
        statement = statement.where(Track.bpm == None)  # noqa: E711
    pending = session.exec(statement.order_by(Track.added_at)).all()

    analysed = 0
    failed = 0
    for track in pending[:limit]:
        path = paths.music / track.relpath
        result = analysis.analyse(
            path,
            ffmpeg=cfg.tools.ffmpeg,
            bpm_range=(cfg.audio.analysis.min_bpm, cfg.audio.analysis.max_bpm),
            prefer_librosa=cfg.audio.analysis.prefer_librosa,
        )
        if result.bpm > 0 and result.confidence >= cfg.audio.analysis.min_confidence:
            track.bpm = round(result.bpm, 2)
            track.beat_offset_s = round(result.beat_offset_s, 4)
            track.bpm_confidence = round(result.confidence, 3)
            track.analysis_method = result.method
            analysed += 1
        else:
            # Record the attempt so it is not retried forever. A zero BPM says
            # "measured, and the answer was not usable".
            track.bpm = 0.0
            track.bpm_confidence = round(result.confidence, 3)
            track.analysis_method = result.method
            failed += 1
        session.add(track)

    outcome = {
        "analysed": analysed,
        "failed": failed,
        "remaining": max(0, len(pending) - limit),
        "skipped": False,
    }
    if analysed or failed:
        log.info("tempo analysis", extra=outcome)
    return outcome


def analysis_summary(session: Session) -> dict[str, object]:
    tracks = session.exec(select(Track).where(Track.missing == False)).all()  # noqa: E712
    measured = [t for t in tracks if t.bpm]
    return {
        "tracks": len(tracks),
        "analysed": len(measured),
        "pending": sum(1 for t in tracks if t.bpm is None),
        "unusable": sum(1 for t in tracks if t.bpm == 0.0),
        "mean_bpm": round(sum(t.bpm for t in measured) / len(measured), 1) if measured else None,
        "librosa": analysis.librosa_available(),
    }


def find_duplicates(session: Session) -> dict[str, list[Track]]:
    """Group tracks by content hash, returning only the groups with more than one."""
    groups: dict[str, list[Track]] = {}
    for track in session.exec(select(Track).where(Track.missing == False)).all():  # noqa: E712
        if track.sha256:
            groups.setdefault(track.sha256, []).append(track)
    return {digest: tracks for digest, tracks in groups.items() if len(tracks) > 1}


def total_duration(session: Session) -> float:
    return sum(
        t.duration_s
        for t in session.exec(select(Track).where(Track.missing == False)).all()  # noqa: E712
    )


__all__ = [
    "SUPPORTED_SUFFIXES",
    "ScanResult",
    "TrackTags",
    "analyse_pending",
    "analysis_summary",
    "content_hash",
    "find_duplicates",
    "music_files",
    "read_tags",
    "scan",
    "total_duration",
]
