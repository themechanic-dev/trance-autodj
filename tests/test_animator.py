"""Clip rendering: sizing, the ffmpeg filter chain, and loop handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Config, load_config
from app.services.visual.animator import (
    build_filter_chain,
    frames_for,
    loop_blend_frames,
    render_size,
)
from app.services.visual.encoder import EncodeProfile

PROFILE = EncodeProfile._build(Config(), "libx264")


def test_render_size_follows_the_scale_and_stays_even():
    cfg = Config.model_validate({"visual": {"procedural": {"render_scale": 0.5}}})
    assert render_size(cfg) == (640, 360)

    # Odd results would be rejected by x264 under yuv420p chroma subsampling.
    cfg = Config.model_validate({"visual": {"procedural": {"render_scale": 0.333}}})
    width, height = render_size(cfg)
    assert width % 2 == 0 and height % 2 == 0


def test_render_size_never_collapses():
    cfg = Config.model_validate({"visual": {"procedural": {"render_scale": 0.001}}})
    width, height = render_size(cfg)
    assert width >= 64 and height >= 64


def test_filter_chain_scales_before_it_finishes():
    """Grain must be applied at output resolution or it becomes smeared blocks."""
    cfg = Config()
    chain = build_filter_chain(cfg, PROFILE)
    parts = chain.split(",")
    assert parts[0].startswith("scale=1280:720")
    assert parts[-1] == "format=yuv420p"
    assert parts.index("vignette=PI/5") > 0


def test_effects_can_be_turned_off_individually():
    cfg = Config.model_validate(
        {"visual": {"clip": {"film_grain": 0.0, "vignette": False, "chromatic_aberration": 0.0}}}
    )
    chain = build_filter_chain(cfg, PROFILE)
    assert "noise" not in chain
    assert "vignette" not in chain
    assert "rgbashift" not in chain
    assert chain.startswith("scale=")


def test_chromatic_aberration_is_measured_in_output_pixels():
    cfg = Config.model_validate({"visual": {"clip": {"chromatic_aberration": 0.0015}}})
    chain = build_filter_chain(cfg, PROFILE)
    # 0.0015 * 1280 rounds to 2
    assert "rgbashift=rh=-2:bh=2" in chain


def test_grain_strength_is_within_the_filter_range():
    cfg = Config.model_validate({"visual": {"clip": {"film_grain": 1.0}}})
    assert "noise=alls=100" in build_filter_chain(cfg, PROFILE)


def test_frames_for_rounds_up():
    assert frames_for(1.0, 30) == 30
    assert frames_for(1.01, 30) == 31
    assert frames_for(0.0, 30) == 1


def test_loop_blend_is_bounded_by_both_rules():
    # One second at 30 fps, when the clip is long enough for it.
    assert loop_blend_frames(900, 30) == 30
    # ...but never more than a quarter of a short clip.
    assert loop_blend_frames(40, 30) == 10
    assert loop_blend_frames(2, 30) >= 1


@pytest.mark.slow
def test_render_a_real_clip(tmp_path: Path):
    """End to end through ffmpeg. Needs ffmpeg on PATH."""
    import shutil

    from app.services.visual.animator import render_clip
    from app.services.visual.encoder import probe
    from app.services.visual.palettes import Palette
    from app.services.visual.procedural import get

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    cfg = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "data"),
            "TAD_VIDEO__WIDTH": "320",
            "TAD_VIDEO__HEIGHT": "180",
        },
    )
    profile = EncodeProfile._build(cfg, "libx264")
    palette = Palette(id="t", name="T", colors=("#000000", "#22e0ff", "#ffffff"))

    clip = render_clip(
        get("plasma"),
        cfg=cfg,
        profile=profile,
        palette=palette,
        duration_s=1.0,
        seed=7,
        out_dir=tmp_path / "clips",
    )
    assert clip.path.is_file()
    assert clip.path.stat().st_size > 0
    assert clip.duration_s == pytest.approx(1.0, abs=0.3)

    stream = next(s for s in probe(clip.path)["streams"] if s["codec_type"] == "video")
    assert (stream["width"], stream["height"]) == (320, 180)
    assert stream["pix_fmt"] == "yuv420p"
