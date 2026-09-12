"""The `annotate:` fields that carry measured tempo into Liquidsoap."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import select

from app.core.config import load_config
from app.core.db import create_tables, dispose, init_engine, session_scope
from app.core.paths import Paths
from app.models.entities import Track
from app.services.audio import library, playlists
from app.services.audio.templates import render_liquidsoap

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@pytest.fixture
def env(tmp_path: Path):
    def build(**environ: str):
        cfg = load_config(
            None,
            environ={
                "TAD_APP__DATA_DIR": str(tmp_path / "data"),
                "TAD_APP__CONFIG_DIR": str(CONFIG_DIR),
                **environ,
            },
        )
        paths = Paths.from_config(cfg)
        paths.ensure()
        return cfg, paths

    _, paths = build()
    init_engine(paths.db_file)
    create_tables()
    yield build
    dispose()


def add_track(paths: Paths, name: str, *, bpm: float | None, offset: float = 0.1) -> None:
    (paths.music / name).write_bytes(b"x" * 4096)
    with session_scope() as session:
        library.scan(session, paths)
        track = session.exec(select(Track).where(Track.relpath == name)).one()
        track.bpm = bpm
        track.beat_offset_s = offset
        track.bpm_confidence = 0.9 if bpm else 0.0
        track.analysis_method = "test" if bpm else ""
        session.add(track)


def test_without_beat_alignment_no_crossfade_is_overridden(env):
    """The measured tempo is still passed along — it is useful on its own,
    and it reaches the dashboard through now-playing — but nothing about the
    crossfade changes until alignment is switched on."""
    cfg, paths = env()
    add_track(paths, "a.mp3", bpm=140.0)
    with session_scope() as session:
        playlists.ensure_default(session, paths, cfg)
    text = paths.active_playlist.read_text(encoding="utf-8")
    assert "liq_cross_duration" not in text
    assert "liq_cue_in" not in text
    assert 'bpm="140.00"' in text
    assert str(paths.music / "a.mp3") in text


def test_a_track_with_no_tempo_is_not_annotated_at_all(env):
    cfg, paths = env()
    add_track(paths, "a.mp3", bpm=None)
    with session_scope() as session:
        playlists.ensure_default(session, paths, cfg)
    text = paths.active_playlist.read_text(encoding="utf-8")
    assert "annotate:" not in text


def test_with_beat_alignment_each_track_carries_its_own_crossfade(env):
    cfg, paths = env(
        TAD_AUDIO__CROSSFADE__BEAT_ALIGNED="true",
        TAD_AUDIO__CROSSFADE__DURATION_S="14",
    )
    add_track(paths, "a.mp3", bpm=140.0)
    with session_scope() as session:
        playlists.ensure_default(session, paths, cfg)

    line = next(
        ln
        for ln in paths.active_playlist.read_text(encoding="utf-8").splitlines()
        if ln.startswith("annotate:")
    )
    assert 'bpm="140.00"' in line
    assert "liq_cross_duration=" in line
    assert 'bars="8"' in line  # 8 bars at 140 BPM is 13.7s, the nearest to 14

    duration = float(line.split('liq_cross_duration="')[1].split('"')[0])
    bar = (60.0 / 140.0) * 4
    # Written to three decimals, so allow a millisecond of rounding.
    assert duration / bar == pytest.approx(round(duration / bar), abs=1e-3)


def test_an_unmeasured_track_gets_no_crossfade_override(env):
    cfg, paths = env(TAD_AUDIO__CROSSFADE__BEAT_ALIGNED="true")
    add_track(paths, "a.mp3", bpm=None)
    with session_scope() as session:
        playlists.ensure_default(session, paths, cfg)
    text = paths.active_playlist.read_text(encoding="utf-8")
    assert "liq_cross_duration" not in text


def test_cue_in_is_separate_from_bar_alignment(env):
    """Trimming audio off the front is a bigger decision than choosing a
    crossfade length, so it needs its own switch."""
    cfg, paths = env(TAD_AUDIO__CROSSFADE__BEAT_ALIGNED="true")
    add_track(paths, "a.mp3", bpm=140.0, offset=0.15)
    with session_scope() as session:
        playlists.ensure_default(session, paths, cfg)
    assert "liq_cue_in" not in paths.active_playlist.read_text(encoding="utf-8")

    cfg, paths = env(
        TAD_AUDIO__CROSSFADE__BEAT_ALIGNED="true",
        TAD_AUDIO__CROSSFADE__BEAT_ALIGN_CUE_IN="true",
    )
    with session_scope() as session:
        playlists.write_m3u(session, playlists.active(session), paths, cfg)
    assert 'liq_cue_in="0.150"' in paths.active_playlist.read_text(encoding="utf-8")


def test_a_late_first_beat_is_not_trimmed(env):
    """max_cue_in_s stops it cutting into a long intro."""
    cfg, paths = env(
        TAD_AUDIO__CROSSFADE__BEAT_ALIGNED="true",
        TAD_AUDIO__CROSSFADE__BEAT_ALIGN_CUE_IN="true",
        TAD_AUDIO__CROSSFADE__MAX_CUE_IN_S="0.1",
    )
    add_track(paths, "a.mp3", bpm=140.0, offset=1.5)
    with session_scope() as session:
        playlists.ensure_default(session, paths, cfg)
    assert "liq_cue_in" not in paths.active_playlist.read_text(encoding="utf-8")


def test_annotation_values_are_quoted_and_the_path_is_last(env):
    cfg, paths = env(TAD_AUDIO__CROSSFADE__BEAT_ALIGNED="true")
    add_track(paths, "a.mp3", bpm=138.0)
    with session_scope() as session:
        playlists.ensure_default(session, paths, cfg)
    line = next(
        ln
        for ln in paths.active_playlist.read_text(encoding="utf-8").splitlines()
        if ln.startswith("annotate:")
    )
    assert line.endswith(str(paths.music / "a.mp3"))
    fields = line[len("annotate:") :].rsplit(":", 1)[0]
    for field in fields.split(","):
        key, _, value = field.partition("=")
        assert key and value.startswith('"') and value.endswith('"')


# --- the generated script ------------------------------------------------


def test_the_script_only_asks_for_overrides_when_alignment_is_on(env):
    cfg, paths = env()
    plain = render_liquidsoap(cfg, paths, source_password="x")
    assert "override_duration" not in plain
    assert "cue_in_metadata" not in plain

    cfg, paths = env(
        TAD_AUDIO__CROSSFADE__BEAT_ALIGNED="true",
        TAD_AUDIO__CROSSFADE__BEAT_ALIGN_CUE_IN="true",
    )
    aligned = render_liquidsoap(cfg, paths, source_password="x")
    assert 'override_duration="liq_cross_duration"' in aligned
    assert 'cue_in_metadata="liq_cue_in"' in aligned
    assert "crossfade_seconds" in aligned


def test_cue_in_needs_alignment_to_be_on_as_well(env):
    """Asking to cue without aligning would trim audio for no reason."""
    cfg, paths = env(TAD_AUDIO__CROSSFADE__BEAT_ALIGN_CUE_IN="true")
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert "cue_in_metadata" not in script


def test_the_transition_takes_the_shorter_of_the_two_sides(env):
    """Fading out over eight bars while fading in over four leaves a hole."""
    cfg, paths = env(TAD_AUDIO__CROSSFADE__BEAT_ALIGNED="true")
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert "min(crossfade_seconds(a.metadata), crossfade_seconds(b.metadata))" in script
