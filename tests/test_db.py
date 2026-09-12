"""The schema must actually create, and its constraints must hold."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.core.db import create_tables, dispose, init_engine, session_scope
from app.models import Block, BlockStatus, Playlist, PlaylistItem, Track


@pytest.fixture
def db(tmp_path: Path):
    init_engine(tmp_path / "test.sqlite")
    create_tables()
    yield
    dispose()


def test_tables_are_created(db):
    with session_scope() as session:
        assert session.exec(select(Track)).all() == []
        assert session.exec(select(Block)).all() == []


def test_wal_is_enabled(db, tmp_path: Path):
    with session_scope() as session:
        mode = session.connection().exec_driver_sql("PRAGMA journal_mode").scalar()
    assert str(mode).lower() == "wal"


def test_track_relpath_is_unique(db):
    with session_scope() as session:
        session.add(Track(relpath="a/b.mp3", title="One"))
    with pytest.raises(IntegrityError), session_scope() as session:
        session.add(Track(relpath="a/b.mp3", title="Two"))


def test_playlist_positions_are_unique_within_a_playlist(db):
    with session_scope() as session:
        track_a = Track(relpath="1.mp3")
        track_b = Track(relpath="2.mp3")
        playlist = Playlist(name="Night")
        session.add_all([track_a, track_b, playlist])
        session.flush()
        session.add_all(
            [
                PlaylistItem(playlist_id=playlist.id, track_id=track_a.id, position=0),
                PlaylistItem(playlist_id=playlist.id, track_id=track_b.id, position=1),
            ]
        )

    with pytest.raises(IntegrityError), session_scope() as session:
        playlist = session.exec(select(Playlist)).one()
        track = session.exec(select(Track)).first()
        session.add(PlaylistItem(playlist_id=playlist.id, track_id=track.id, position=0))


def test_deleting_a_playlist_removes_its_items(db):
    with session_scope() as session:
        track = Track(relpath="1.mp3")
        playlist = Playlist(name="Day")
        session.add_all([track, playlist])
        session.flush()
        session.add(PlaylistItem(playlist_id=playlist.id, track_id=track.id, position=0))

    with session_scope() as session:
        session.delete(session.exec(select(Playlist)).one())

    with session_scope() as session:
        assert session.exec(select(PlaylistItem)).all() == []
        # The track itself survives; only the membership went away.
        assert len(session.exec(select(Track)).all()) == 1


def test_block_defaults(db):
    with session_scope() as session:
        session.add(Block(id="abc", relpath="blocks/abc.ts"))
    with session_scope() as session:
        block = session.exec(select(Block)).one()
        assert block.status is BlockStatus.BUILDING
        assert block.play_count == 0
        assert block.pinned is False
        assert block.created_at is not None


def test_rollback_on_error_leaves_nothing_behind(db):
    with pytest.raises(RuntimeError), session_scope() as session:
        session.add(Track(relpath="ghost.mp3"))
        session.flush()
        raise RuntimeError("boom")
    with session_scope() as session:
        assert session.exec(select(Track)).all() == []
