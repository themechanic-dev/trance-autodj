"""The play log: what was on air, and when.

A copyright notice arrives with a timestamp and points at the whole library.
These tests are about the one question that narrows it to a file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.core.config import load_config
from app.core.db import create_tables, dispose, init_engine, session_scope
from app.core.paths import Paths
from app.models.entities import PlayLogEntry, Track
from app.services.audio import playlog


@pytest.fixture
def env(tmp_path: Path):
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "data")})
    paths = Paths.from_config(cfg)
    paths.ensure()
    init_engine(paths.db_file)
    create_tables()
    yield cfg, paths
    dispose()


def playing(paths: Paths, relpath: str, *, started: float, title: str = "", artist: str = ""):
    paths.nowplaying_file.write_text(
        json.dumps(
            {
                "filename": str(paths.music / relpath),
                "title": title or relpath,
                "artist": artist,
                "started_at": started,
            }
        ),
        encoding="utf-8",
    )


def test_a_track_change_is_written_down(env):
    _, paths = env
    playing(paths, "one.mp3", started=1000.0, title="One")
    with session_scope() as session:
        assert playlog.record(session, paths) is not None
    with session_scope() as session:
        entries = playlog.recent(session)
        assert [e.relpath for e in entries] == ["one.mp3"]
        assert entries[0].ended_at is None, "the track that is playing has not ended"


def test_the_same_track_is_not_written_twice(env):
    """The recorder polls every few seconds; a track is minutes long."""
    _, paths = env
    playing(paths, "one.mp3", started=1000.0)
    with session_scope() as session:
        playlog.record(session, paths)
    for _ in range(5):
        with session_scope() as session:
            assert playlog.record(session, paths) is None
    with session_scope() as session:
        assert len(playlog.recent(session)) == 1


def test_the_previous_entry_is_closed_when_the_next_begins(env):
    _, paths = env
    playing(paths, "one.mp3", started=1000.0)
    with session_scope() as session:
        playlog.record(session, paths, when=datetime(2026, 9, 12, 20, 0, tzinfo=UTC))

    playing(paths, "two.mp3", started=1300.0)
    closed_at = datetime(2026, 9, 12, 20, 5, tzinfo=UTC)
    with session_scope() as session:
        playlog.record(session, paths, when=closed_at)

    with session_scope() as session:
        entries = {e.relpath: e for e in playlog.recent(session)}
        assert entries["one.mp3"].ended_at is not None
        assert entries["two.mp3"].ended_at is None


def test_the_station_going_quiet_closes_the_entry(env):
    """Otherwise the log claims a track is still playing hours later, and the
    answer to 'what was on at three in the morning' is a lie."""
    _, paths = env
    playing(paths, "one.mp3", started=1000.0)
    with session_scope() as session:
        playlog.record(session, paths)

    paths.nowplaying_file.unlink()
    with session_scope() as session:
        assert playlog.record(session, paths) is None
    with session_scope() as session:
        assert playlog.recent(session)[0].ended_at is not None


def test_what_was_playing_at_a_given_moment(env):
    """The whole point."""
    base = datetime(2026, 9, 12, 22, 0, tzinfo=UTC)
    with session_scope() as session:
        session.add(
            PlayLogEntry(
                relpath="early.mp3",
                title="Early",
                started_at=base,
                ended_at=base + timedelta(minutes=6),
            )
        )
        session.add(
            PlayLogEntry(
                relpath="later.mp3",
                title="Later",
                started_at=base + timedelta(minutes=6),
                ended_at=base + timedelta(minutes=12),
            )
        )

    with session_scope() as session:
        assert playlog.at(session, base + timedelta(minutes=3)).relpath == "early.mp3"
        assert playlog.at(session, base + timedelta(minutes=9)).relpath == "later.mp3"
        assert playlog.at(session, base - timedelta(minutes=1)) is None


def test_the_record_survives_the_track_being_deleted(env):
    """The operator deletes the claimed file the moment they find out which
    one it was — and the log still has to say what happened."""
    with session_scope() as session:
        track = Track(relpath="claimed.mp3", title="Claimed")
        session.add(track)
        session.flush()
        session.add(
            PlayLogEntry(
                track_id=track.id,
                relpath="claimed.mp3",
                title="Claimed",
                started_at=datetime(2026, 9, 12, 23, 14, tzinfo=UTC),
            )
        )

    with session_scope() as session:
        session.delete(session.exec(_tracks()).first())

    with session_scope() as session:
        entry = playlog.recent(session)[0]
        assert entry.relpath == "claimed.mp3", "the name went with the file"
        assert entry.title == "Claimed"
        assert entry.track_id is None


def _tracks():
    from sqlmodel import select

    return select(Track)


def test_a_time_typed_by_a_person_is_read_in_station_time():
    """A claim is read off a screen in local time. Asking the operator to
    convert to UTC is asking at the worst possible moment."""
    athens = playlog.parse_moment("2026-09-12T23:14", "Europe/Athens")
    assert athens.hour == 20, "23:14 in Athens is 20:14 UTC in September"

    explicit = playlog.parse_moment("2026-09-12T23:14:00+00:00", "Europe/Athens")
    assert explicit.hour == 23, "a time that states its zone is believed"


def test_a_nonsense_timezone_does_not_lose_the_answer():
    moment = playlog.parse_moment("2026-09-12T23:14", "Mars/Olympus_Mons")
    assert moment.hour == 23


def test_the_log_does_not_grow_without_end(env):
    base = datetime(2026, 9, 12, tzinfo=UTC)
    with session_scope() as session:
        for index in range(20):
            session.add(
                PlayLogEntry(relpath=f"{index}.mp3", started_at=base + timedelta(minutes=index))
            )

    with session_scope() as session:
        assert playlog.prune(session, keep=12) == 8
    with session_scope() as session:
        left = playlog.recent(session, limit=100)
        assert len(left) == 12
        assert "0.mp3" not in {e.relpath for e in left}, "the oldest should have gone first"


def test_pruning_a_short_log_does_nothing(env):
    with session_scope() as session:
        assert playlog.prune(session, keep=10) == 0


def test_the_recorder_starts_and_stops(env):
    _, paths = env
    recorder = playlog.Recorder(paths)
    assert not recorder.running
    recorder.start()
    assert recorder.running
    recorder.stop()
    assert not recorder.running
