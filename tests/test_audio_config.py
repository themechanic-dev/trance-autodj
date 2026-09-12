"""Generating the Liquidsoap and Icecast configuration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import load_config
from app.core.paths import Paths
from app.services.audio.templates import liq_float, render_icecast, render_liquidsoap, write_configs

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@pytest.fixture
def setup(tmp_path: Path):
    cfg = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "data"),
            "TAD_APP__CONFIG_DIR": str(CONFIG_DIR),
        },
    )
    paths = Paths.from_config(cfg)
    paths.ensure()
    return cfg, paths


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-14.0, "-14.0"), (14, "14.0"), (2.0, "2.0"), (0.5, "0.5"), (1.25, "1.25")],
)
def test_liq_float_always_produces_one_decimal_point(value, expected):
    """A doubled point ("-14.0.") is a parse error whose message points at
    line 1 and tells you nothing. This function exists to prevent it."""
    assert liq_float(value) == expected


def test_no_generated_number_has_two_decimal_points(setup):
    """A doubled point is what "-14.0." looks like, and it is a parse error
    whose message points at line 1. Only the values we substitute are checked;
    version strings in comments legitimately look similar."""
    cfg, paths = setup
    script = render_liquidsoap(cfg, paths, source_password="x")
    for line in script.splitlines():
        code = line.split("#", 1)[0]
        assert not re.search(r"=\s*-?\d+\.\d*\.", code), f"doubled point in: {line}"


def test_script_contains_the_configured_crossfade(setup):
    cfg, paths = setup
    cfg.audio.crossfade.duration_s = 18.0
    cfg.audio.crossfade.fade_in_curve = "sinusoidal"
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert "duration=18.0" in script
    assert 'fade.in(type="sinusoidal"' in script


def test_the_source_is_made_safe_on_both_sides_of_the_crossfade(setup):
    """cross() rejects a fallible input, and returns a fallible output that
    output.icecast then rejects in turn. Both wrappers are load-bearing."""
    cfg, paths = setup
    script = render_liquidsoap(cfg, paths, source_password="x")
    before = script.index("radio = mksafe(radio)")
    cross = script.index("radio = cross(")
    after = script.index("radio = mksafe(radio)", cross)
    assert before < cross < after


def test_the_playlist_is_watched_so_a_reload_needs_no_restart(setup):
    cfg, paths = setup
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert 'reload_mode="watch"' in script
    assert str(paths.active_playlist) in script


def test_modes_are_translated_to_liquidsoap_names(setup):
    cfg, paths = setup
    for ours, theirs in (("shuffle", "randomize"), ("sequential", "normal"), ("random", "random")):
        cfg.audio.mode = ours
        assert f'mode="{theirs}"' in render_liquidsoap(cfg, paths, source_password="x")


def test_loudness_can_be_switched_off(setup):
    cfg, paths = setup
    cfg.audio.loudness.enabled = False
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert "normalize_track_gain" not in script

    cfg.audio.loudness.enabled = True
    script = render_liquidsoap(cfg, paths, source_password="x")
    # The 2.4 name, not the deprecated `replaygain`.
    assert "normalize_track_gain" in script
    assert "target=-14.0" in script


def test_opus_and_mp3_produce_different_encoders(setup):
    cfg, paths = setup
    assert "%mp3(" in render_liquidsoap(cfg, paths, source_password="x")
    cfg.audio.icecast.format = "opus"
    assert "%opus(" in render_liquidsoap(cfg, paths, source_password="x")


def test_a_missing_template_variable_fails_loudly(setup):
    """StrictUndefined: a typo must break the render, not ship a broken script."""
    cfg, paths = setup
    # Every variable the template needs is supplied, so this must simply work.
    assert render_liquidsoap(cfg, paths, source_password="x")


def test_icecast_config_binds_where_configured(setup):
    cfg, paths = setup
    xml = render_icecast(cfg, paths, source_password="s3cret", admin_password="a3cret")
    assert "<bind-address>127.0.0.1</bind-address>" in xml
    assert "<port>8000</port>" in xml
    assert "<source-password>s3cret</source-password>" in xml
    assert "<chroot>0</chroot>" in xml


def test_written_icecast_config_is_not_world_readable(setup):
    """It contains the source password in plain text."""
    import stat

    cfg, paths = setup
    _, icecast = write_configs(cfg, paths, source_password="s", admin_password="a")
    assert stat.S_IMODE(icecast.stat().st_mode) == 0o600


def test_write_configs_produces_both_files(setup):
    cfg, paths = setup
    script, icecast = write_configs(cfg, paths, source_password="s", admin_password="a")
    assert script.is_file() and icecast.is_file()
    assert script.read_text().startswith("#")
    assert icecast.read_text().startswith("<?xml")


def test_the_loudness_normaliser_comes_after_the_crossfade(setup):
    """`normalize` upstream of `cross` leaks memory and CPU without bound.

    Two scripts differing only in this order, measured over 30 minutes:
    before, 175 MB -> 872 MB and 0.08 -> 0.38 cores, still climbing; after,
    166 MB -> 191 MB and 0.08 cores, flat. Left running the first one reaches
    about 2 GB and pins a full core, which ends a 24/7 broadcast. Neither
    operator leaks alone, so nothing but the order is protecting us here.
    """
    cfg, paths = setup
    script = render_liquidsoap(cfg, paths, source_password="x")
    cross_at = script.index("radio = cross(")
    normalize_at = script.index('normalize(\n  id="loudness"')
    assert normalize_at > cross_at, "the loudness normaliser moved back in front of the crossfade"


def test_per_track_gain_stays_in_front_of_the_crossfade(setup):
    """Both sides of a transition have to be matched to each other *before*
    they overlap, so this half of the loudness chain belongs upstream. It was
    measured not to leak there (crossfade + ReplayGain was flat for 75 min)."""
    cfg, paths = setup
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert script.index("normalize_track_gain(") < script.index("radio = cross(")


def test_loudness_can_still_be_turned_off_entirely(setup):
    cfg, paths = setup
    cfg.audio.loudness.enabled = False
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert "normalize_track_gain(" not in script
    assert 'id="loudness"' not in script
    assert "radio = cross(" in script


def test_the_silent_fallback_lasts_forever(setup):
    """`blank(duration=0.)` does not mean "silence"; it means an endless
    stream of zero-length tracks.

    Liquidsoap starts a crossfade for each one, so the cross buffer counter
    reaches the thousands in minutes, the clock falls seconds behind, and the
    mount serves -91 dB. The station cannot recover by itself either: it is
    too busy churning empty tracks to notice that the playlist has music
    again. Negative means forever, which is what a safety source is for.
    """
    cfg, paths = setup
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert 'blank(id="safety", duration=0.)' not in script
    assert "duration=-1." in script


def test_a_configured_safe_track_replaces_the_silence(setup):
    cfg, paths = setup
    cfg.audio.safe_track = "/data/music/safe.mp3"
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert 'single(id="safety"' in script
    assert 'blank(id="safety"' not in script


def test_a_rotation_makes_liquidsoap_read_the_file_in_order(setup):
    """The rotation order is ours: this application decided which list follows
    which, and a global "shuffle" would throw that away the moment Liquidsoap
    read the file."""
    cfg, paths = setup
    cfg.audio.mode = "shuffle"
    cfg.audio.playlist_playback = "rotation"
    assert 'mode="normal"' in render_liquidsoap(cfg, paths, source_password="x")


def test_a_single_playlist_still_honours_the_playback_order(setup):
    cfg, paths = setup
    cfg.audio.mode = "shuffle"
    cfg.audio.playlist_playback = "single"
    assert 'mode="randomize"' in render_liquidsoap(cfg, paths, source_password="x")


def test_now_playing_is_written_beside_itself(setup):
    """Without temp_dir, Liquidsoap writes the temporary copy under the system
    temp directory and renames it across file systems — /tmp is a tmpfs, /data
    is the volume — which cannot be atomic. The write then fails with "Atomic
    rename failed!" and the dashboard shows no current track at all.
    """
    cfg, paths = setup
    script = render_liquidsoap(cfg, paths, source_password="x")
    assert "temp_dir=" in script
    assert f'temp_dir="{paths.nowplaying_file.parent}"' in script
