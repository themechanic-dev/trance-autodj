"""The music library and playlists."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import select

from app.core.config import load_config
from app.core.db import create_tables, dispose, init_engine, session_scope
from app.core.paths import Paths
from app.models.entities import Playlist, Track
from app.services.audio import library, playlists


@pytest.fixture
def env(tmp_path: Path):
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "data")})
    paths = Paths.from_config(cfg)
    paths.ensure()
    init_engine(paths.db_file)
    create_tables()
    yield cfg, paths
    dispose()


def make_file(paths: Paths, name: str, content: bytes = b"x" * 5000) -> Path:
    path = paths.music / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_only_audio_extensions_are_picked_up(env):
    _, paths = env
    make_file(paths, "a.mp3")
    make_file(paths, "b.flac")
    make_file(paths, "notes.txt")
    make_file(paths, "cover.jpg")
    found = {p.name for p in library.music_files(paths.music)}
    assert found == {"a.mp3", "b.flac"}


def test_nested_directories_are_scanned(env):
    _, paths = env
    make_file(paths, "sets/2026/track.mp3")
    with session_scope() as session:
        assert library.scan(session, paths).added == 1
    with session_scope() as session:
        assert session.exec(select(Track)).one().relpath == "sets/2026/track.mp3"


def test_a_deleted_file_marks_the_row_missing_and_keeps_it(env):
    """An unmounted disk must cost a rescan, not the library."""
    _, paths = env
    path = make_file(paths, "gone.mp3")
    with session_scope() as session:
        library.scan(session, paths)
    path.unlink()
    with session_scope() as session:
        assert library.scan(session, paths).missing == 1
    with session_scope() as session:
        track = session.exec(select(Track)).one()
        assert track.missing is True


def test_a_returning_file_is_restored(env):
    _, paths = env
    path = make_file(paths, "back.mp3")
    with session_scope() as session:
        library.scan(session, paths)
    path.unlink()
    with session_scope() as session:
        library.scan(session, paths)
    make_file(paths, "back.mp3")
    with session_scope() as session:
        assert library.scan(session, paths).restored == 1


def test_unchanged_files_are_skipped_on_rescan(env):
    _, paths = env
    make_file(paths, "a.mp3")
    with session_scope() as session:
        library.scan(session, paths)
    with session_scope() as session:
        result = library.scan(session, paths)
    assert result.skipped == 1
    assert result.added == 0


def test_identical_content_is_reported_as_a_duplicate(env):
    _, paths = env
    make_file(paths, "one.mp3", b"same-bytes" * 500)
    make_file(paths, "two.mp3", b"same-bytes" * 500)
    make_file(paths, "other.mp3", b"different" * 500)
    with session_scope() as session:
        library.scan(session, paths)
    with session_scope() as session:
        groups = library.find_duplicates(session)
    assert len(groups) == 1
    assert {t.relpath for t in next(iter(groups.values()))} == {"one.mp3", "two.mp3"}


def test_content_hash_differs_for_different_files(env):
    _, paths = env
    a = make_file(paths, "a.mp3", b"aaaa" * 1000)
    b = make_file(paths, "b.mp3", b"bbbb" * 1000)
    assert library.content_hash(a) != library.content_hash(b)


def test_a_file_with_unreadable_tags_still_gets_a_title(env):
    """A file mutagen cannot parse is a playable file with an ugly name, not
    a nameless row nobody can find in the library."""
    _, paths = env
    path = make_file(paths, "Untagged Track.mp3")
    assert library.read_tags(path).title == "Untagged Track"


# --- playlists -----------------------------------------------------------


def test_the_m3u_is_extended_and_absolute(env):
    _, paths = env
    make_file(paths, "a.mp3")
    with session_scope() as session:
        library.scan(session, paths)
        playlists.ensure_default(session, paths)

    text = paths.active_playlist.read_text(encoding="utf-8")
    assert text.startswith("#EXTM3U")
    assert "#EXTINF:" in text
    assert str(paths.music / "a.mp3") in text


def test_the_automatic_playlist_follows_the_library(env):
    """It used to be created empty at startup and never refilled, so the
    station played silence even with music on disk."""
    _, paths = env
    with session_scope() as session:
        playlists.ensure_default(session, paths)  # nothing on disk yet
    with session_scope() as session:
        assert playlists.summarise(session, playlists.active(session)).track_count == 0

    make_file(paths, "a.mp3")
    make_file(paths, "b.mp3")
    with session_scope() as session:
        library.scan(session, paths)
        playlists.ensure_default(session, paths)
    with session_scope() as session:
        assert playlists.summarise(session, playlists.active(session)).track_count == 2


def test_the_automatic_playlist_survives_a_second_change(env):
    """Adding music and scanning again used to raise InvalidRequestError:
    "Instance '<PlaylistItem>' has been deleted".

    ensure_default reads playlist.items to decide whether anything changed,
    set_tracks then deletes those very rows, and the save that follows
    cascades back into the deleted instances. It only appears on the *second*
    change, because the first has nothing to delete — which is how it reached
    a finished build: every test and the demo volume grew the library once.
    """
    _, paths = env
    make_file(paths, "a.mp3")
    with session_scope() as session:
        library.scan(session, paths)
        playlists.ensure_default(session, paths)
    with session_scope() as session:
        assert playlists.summarise(session, playlists.active(session)).track_count == 1

    make_file(paths, "b.mp3", b"y" * 5000)
    make_file(paths, "c.mp3", b"z" * 5000)
    with session_scope() as session:
        library.scan(session, paths)
        playlists.ensure_default(session, paths)
    with session_scope() as session:
        assert playlists.summarise(session, playlists.active(session)).track_count == 3


def test_the_automatic_playlist_shrinks_too(env):
    """The same path, in the direction that deletes more than it adds."""
    _, paths = env
    for name in ("a.mp3", "b.mp3", "c.mp3"):
        make_file(paths, name, name.encode() * 1000)
    with session_scope() as session:
        library.scan(session, paths)
        playlists.ensure_default(session, paths)

    (paths.music / "b.mp3").unlink()
    (paths.music / "c.mp3").unlink()
    with session_scope() as session:
        library.scan(session, paths)
        playlists.ensure_default(session, paths)
    with session_scope() as session:
        assert playlists.summarise(session, playlists.active(session)).track_count == 1


def test_a_hand_made_active_playlist_is_left_alone(env):
    _, paths = env
    make_file(paths, "a.mp3")
    with session_scope() as session:
        library.scan(session, paths)
        mine = playlists.create(session, "Saturday night")
        playlists.activate(session, mine, paths)

    with session_scope() as session:
        playlists.ensure_default(session, paths)
    with session_scope() as session:
        active = playlists.active(session)
        assert active.name == "Saturday night"
        assert playlists.summarise(session, active).track_count == 0


def test_activating_deactivates_the_previous_one(env):
    _, paths = env
    with session_scope() as session:
        first = playlists.create(session, "one")
        second = playlists.create(session, "two")
        playlists.activate(session, first, paths)
        playlists.activate(session, second, paths)
    with session_scope() as session:
        active = session.exec(select(Playlist).where(Playlist.is_active)).all()
        assert [p.name for p in active] == ["two"]


def test_set_tracks_preserves_the_given_order(env):
    _, paths = env
    for name in ("c.mp3", "a.mp3", "b.mp3"):
        make_file(paths, name)
    with session_scope() as session:
        library.scan(session, paths)
        tracks = {t.relpath: t.id for t in session.exec(select(Track)).all()}
        playlist = playlists.create(session, "ordered")
        wanted = [tracks["b.mp3"], tracks["c.mp3"], tracks["a.mp3"]]
        playlists.set_tracks(session, playlist, wanted)

    with session_scope() as session:
        playlist = session.exec(select(Playlist).where(Playlist.name == "ordered")).one()
        got = [t.relpath for t in playlists.tracks_of(session, playlist)]
    assert got == ["b.mp3", "c.mp3", "a.mp3"]


def test_missing_tracks_are_left_out_of_the_m3u(env):
    _, paths = env
    make_file(paths, "here.mp3")
    path = make_file(paths, "gone.mp3")
    with session_scope() as session:
        library.scan(session, paths)
        playlists.ensure_default(session, paths)
    path.unlink()
    with session_scope() as session:
        library.scan(session, paths)
        playlists.write_m3u(session, playlists.active(session), paths)

    text = paths.active_playlist.read_text(encoding="utf-8")
    assert "here.mp3" in text
    assert "gone.mp3" not in text


def test_unknown_track_ids_are_ignored(env):
    with session_scope() as session:
        playlist = playlists.create(session, "ghosts")
        assert playlists.set_tracks(session, playlist, [999, 1000]) == 0


def test_deleting_a_track_that_is_in_a_playlist_actually_deletes_it(env):
    """It used to answer 200, delete the file, and keep the row.

    Without a cascade on Track.items, SQLAlchemy disowns the playlist entries
    by setting playlist_items.track_id to NULL — a NOT NULL column — so the
    commit fails. The commit runs in the request's teardown, after the
    response has gone out and after the file has been unlinked, so the caller
    is told it worked, the audio really is gone, and the library keeps showing
    a track that no longer exists.
    """
    from app.models.entities import PlaylistItem

    _, paths = env
    make_file(paths, "a.mp3")
    make_file(paths, "b.mp3", b"y" * 5000)
    with session_scope() as session:
        library.scan(session, paths)
        playlists.ensure_default(session, paths)

    with session_scope() as session:
        track = session.exec(select(Track).where(Track.relpath == "a.mp3")).one()
        assert session.exec(
            select(PlaylistItem).where(PlaylistItem.track_id == track.id)
        ).all(), "the fixture is not testing what it claims"
        session.delete(track)

    with session_scope() as session:
        assert session.exec(select(Track).where(Track.relpath == "a.mp3")).first() is None
        assert not session.exec(
            select(PlaylistItem).where(PlaylistItem.track_id == 1)
        ).all(), "the playlist still points at a track that is gone"


def _two_lists(paths):
    """Two playlists over four tracks: a, b in one and c, d in the other."""
    for name in ("a.mp3", "b.mp3", "c.mp3", "d.mp3"):
        make_file(paths, name, name.encode() * 900)
    with session_scope() as session:
        library.scan(session, paths)
        ids = {t.relpath: t.id for t in session.exec(select(Track)).all()}
        first = playlists.create(session, "Opening", mode=playlists.PlaylistMode.SEQUENTIAL)
        second = playlists.create(session, "Peak", mode=playlists.PlaylistMode.SEQUENTIAL)
        session.flush()
        playlists.set_tracks(session, first, [ids["a.mp3"], ids["b.mp3"]])
        playlists.set_tracks(session, second, [ids["c.mp3"], ids["d.mp3"]])
        return first.id, second.id


def _m3u_files(paths):
    return [
        line.rsplit("/", 1)[-1]
        for line in paths.active_playlist.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def test_the_rotation_plays_the_lists_in_the_order_given(env):
    """The whole point: the operator decides which list follows which."""
    cfg, paths = env
    first, second = _two_lists(paths)
    cfg.audio.playlist_playback = "rotation"

    with session_scope() as session:
        playlists.set_rotation(session, [second, first])
        playlists.write_active(session, paths, cfg)
    assert _m3u_files(paths) == ["c.mp3", "d.mp3", "a.mp3", "b.mp3"]

    with session_scope() as session:
        playlists.set_rotation(session, [first, second])
        playlists.write_active(session, paths, cfg)
    assert _m3u_files(paths) == ["a.mp3", "b.mp3", "c.mp3", "d.mp3"]


def test_single_mode_ignores_the_rotation(env):
    """Arranging a rotation while one list is on air must not change the air."""
    cfg, paths = env
    first, second = _two_lists(paths)
    cfg.audio.playlist_playback = "single"

    with session_scope() as session:
        playlist = session.get(playlists.Playlist, first)
        playlists.activate(session, playlist, paths, cfg)
        playlists.set_rotation(session, [second])
        playlists.write_active(session, paths, cfg)
    assert _m3u_files(paths) == ["a.mp3", "b.mp3"]


def test_an_empty_rotation_falls_back_to_the_active_list(env):
    """Handing Liquidsoap an empty file drops the station onto the silent
    safety source, which is the one failure this project cannot afford."""
    cfg, paths = env
    first, _ = _two_lists(paths)
    cfg.audio.playlist_playback = "rotation"
    with session_scope() as session:
        playlists.activate(session, session.get(playlists.Playlist, first), paths, cfg)
        playlists.set_rotation(session, [])
        assert playlists.write_active(session, paths, cfg) == 2
    assert _m3u_files(paths) == ["a.mp3", "b.mp3"]


def test_reordering_never_leaves_gaps_or_duplicates(env):
    _, paths = env
    first, second = _two_lists(paths)
    with session_scope() as session:
        playlists.set_rotation(session, [first, second])
        playlists.set_rotation(session, [second])
        positions = [p.rotation_position for p in playlists.rotation(session)]
        assert positions == [0]
        assert [p.id for p in playlists.rotation(session)] == [second]


def test_wma_files_are_part_of_the_library(env):
    """ffmpeg decodes wmav1/wmav2/wmapro/wmalossless and Liquidsoap decodes
    through ffmpeg, so leaving .wma out was a list, not a limitation — and a
    file the scanner ignores is invisible: no row, no delete button, nothing
    to explain why it never plays."""
    _, paths = env
    make_file(paths, "track.wma")
    make_file(paths, "notes.txt")
    with session_scope() as session:
        library.scan(session, paths)
        names = {t.relpath for t in session.exec(select(Track)).all()}
    assert names == {"track.wma"}


def test_asf_tags_are_read_under_their_own_names(tmp_path: Path):
    """ASF calls the artist "Author" and mutagen's easy interface does not
    normalise it, so a WMA library imported as titles by nobody."""
    import mutagen

    from app.services.audio.library import read_tags

    class _FakeASF:
        tags = {
            "Title": ["Nightfall"],
            "Author": ["Goldenfinger"],
            "WM/AlbumTitle": ["Collected"],
            "WM/Genre": ["Psytrance"],
            "WM/Year": ["2021"],
        }
        info = type("Info", (), {"length": 412.0, "bitrate": 192000, "sample_rate": 44100})()

    path = tmp_path / "x.wma"
    path.write_bytes(b"\0" * 64)
    original = mutagen.File
    mutagen.File = lambda *a, **k: _FakeASF()
    try:
        tags = read_tags(path)
    finally:
        mutagen.File = original

    assert tags.title == "Nightfall"
    assert tags.artist == "Goldenfinger"
    assert tags.album == "Collected"
    assert tags.genre == "Psytrance"
    assert tags.year == 2021
