"""Keeping a record of what went out, and when.

A copyright notice arrives with a timestamp and a thirty-second excerpt. The
station knows what is playing now and how often each track has played, and
neither of those can answer "what was on air at 23:14 last Tuesday" — so the
notice points at a library of several hundred files and nothing narrower.

This is the missing half: one row per track played, opened when it starts and
closed when the next one does. Liquidsoap announces a change by rewriting a
file rather than by calling anything, so something has to watch that file;
that something is a thread here rather than a request handler, because the
answer has to be right whether or not anybody had the dashboard open.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlmodel import Session, select

from app.core.logging import get_logger
from app.core.paths import Paths
from app.models.entities import PlayLogEntry, Track, utcnow
from app.services.audio import nowplaying

log = get_logger(__name__)

#: How often the file is checked. A track is minutes long, so this is far
#: finer than it needs to be — it costs one stat call and buys a start time
#: accurate enough to line up against a claim.
POLL_S = 5.0

#: Roughly three years of a station that never stops. Kept as a row count
#: rather than an age so a station that was off for a month does not lose the
#: month before it.
MAX_ENTRIES = 250_000

#: Pruning walks the table, so it does not happen on every write.
PRUNE_EVERY = 2_000


def as_utc(moment: datetime | None) -> datetime | None:
    """Put the label back on a time that came out of the database.

    SQLite stores no offset, so an aware UTC datetime goes in and a naive one
    comes out holding the same wall clock. Serialising that bare lets a
    browser read it as local time — and answer "what was playing at 23:14"
    with a track from three hours earlier, confidently and wrongly.
    """
    if moment is None:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _naive_utc(moment: datetime) -> datetime:
    """The shape the column actually holds, for comparisons that must match."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return moment.replace(tzinfo=None)


def open_entry(session: Session) -> PlayLogEntry | None:
    """The row for whatever is on air, if anything is."""
    return session.exec(
        select(PlayLogEntry)
        .where(PlayLogEntry.ended_at == None)  # noqa: E711
        .order_by(PlayLogEntry.started_at.desc())  # type: ignore[union-attr]
    ).first()


def record(session: Session, paths: Paths, *, when: datetime | None = None) -> PlayLogEntry | None:
    """Notice a track change and write it down. Returns a new row, or None.

    Idempotent: called twice for the same track it does nothing the second
    time, which is what lets the caller poll as often as it likes.
    """
    playing = nowplaying.read(paths.nowplaying_file)
    moment = when or utcnow()
    current = open_entry(session)

    filename = playing.filename or ""
    if not filename:
        # Nothing on air. Close whatever was open rather than leaving a row
        # that claims a track is still playing hours later.
        if current is not None:
            current.ended_at = moment
            session.add(current)
        return None

    relpath = _relative(filename, paths)
    if current is not None and current.relpath == relpath:
        return None

    if current is not None:
        current.ended_at = moment
        session.add(current)

    track = session.exec(select(Track).where(Track.relpath == relpath)).first()
    started = datetime.fromtimestamp(playing.started_at, tz=UTC) if playing.started_at else moment
    entry = PlayLogEntry(
        track_id=track.id if track else None,
        relpath=relpath,
        title=playing.title or (track.title if track else ""),
        artist=playing.artist or (track.artist if track else ""),
        started_at=started,
    )
    session.add(entry)
    return entry


def _relative(filename: str, paths: Paths) -> str:
    """Liquidsoap reports absolute paths; the library speaks in relative ones."""
    try:
        return str(Path(filename).relative_to(paths.music))
    except ValueError:
        # A file from outside the music folder is still worth recording under
        # whatever name it came with.
        return filename


def at(session: Session, moment: datetime) -> PlayLogEntry | None:
    """What was on air then. The question a claim actually asks."""
    when = _naive_utc(moment)
    return session.exec(
        select(PlayLogEntry)
        .where(PlayLogEntry.started_at <= when)
        .where((PlayLogEntry.ended_at == None) | (PlayLogEntry.ended_at >= when))  # noqa: E711
        .order_by(PlayLogEntry.started_at.desc())  # type: ignore[union-attr]
    ).first()


def recent(session: Session, limit: int = 100) -> list[PlayLogEntry]:
    return list(
        session.exec(
            select(PlayLogEntry)
            .order_by(PlayLogEntry.started_at.desc())  # type: ignore[union-attr]
            .limit(limit)
        ).all()
    )


def parse_moment(raw: str, timezone: str) -> datetime:
    """Read a time the operator typed, in the station's own clock.

    A claim is read off a screen in local time. Storing UTC and asking for UTC
    would make the operator do the arithmetic at the exact moment they are
    least inclined to — so a bare timestamp is taken to mean station time.
    """
    text = raw.strip().replace("Z", "+00:00")
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        try:
            moment = moment.replace(tzinfo=ZoneInfo(timezone))
        except (ZoneInfoNotFoundError, ValueError):
            moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def prune(session: Session, keep: int = MAX_ENTRIES) -> int:
    """Drop the oldest rows once the table is longer than anyone will read."""
    total = len(session.exec(select(PlayLogEntry.id)).all())
    if total <= keep:
        return 0
    doomed = session.exec(
        select(PlayLogEntry)
        .order_by(PlayLogEntry.started_at)  # type: ignore[arg-type]
        .limit(total - keep)
    ).all()
    for entry in doomed:
        session.delete(entry)
    return len(doomed)


class Recorder:
    """Watches the now-playing file and keeps the log honest.

    A thread rather than something hung off a request: the log has to be
    complete on a machine nobody is looking at, which is every machine this
    runs on once it is working.
    """

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writes = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="play-log", daemon=True)
        self._thread.start()
        log.info("play log recorder started")

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        from app.core.db import session_scope

        while not self._stop.is_set():
            try:
                with session_scope() as session:
                    if record(session, self.paths) is not None:
                        self._writes += 1
                        if self._writes % PRUNE_EVERY == 0:
                            prune(session)
            except Exception:
                # A station must not stop because its diary could not be
                # written. Log it and try again on the next tick.
                log.exception("could not update the play log")
            self._stop.wait(POLL_S)


__all__ = [
    "MAX_ENTRIES",
    "POLL_S",
    "Recorder",
    "as_utc",
    "at",
    "open_entry",
    "parse_moment",
    "prune",
    "recent",
    "record",
]
