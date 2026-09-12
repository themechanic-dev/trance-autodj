"""AI stills: the animator, backend selection, and prompt presets."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Config, load_config
from app.core.paths import Paths
from app.services.visual import ai as ai_backends
from app.services.visual.ai.prompts import DEFAULT_NEGATIVE, PromptSet
from app.services.visual.palettes import Palette
from app.services.visual.procedural.base import RenderContext
from app.services.visual.stills import StillAnimator, ease, load_image, loop_position

PALETTE = Palette(id="t", name="T", colors=("#000000", "#ffffff"))
WIDTH, HEIGHT = 96, 64


def synthetic(tmp_path: Path, name: str, tint: tuple[int, int, int]) -> np.ndarray:
    from PIL import Image

    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    field = (np.sin(x / 17) * np.cos(y / 19) + 1) / 2
    data = np.stack([field * c for c in tint], -1).clip(0, 255).astype(np.uint8)
    path = tmp_path / name
    Image.fromarray(data).save(path)
    return load_image(path, WIDTH, HEIGHT)


def context(seed: int = 5, frames: int = 40) -> RenderContext:
    return RenderContext(
        width=WIDTH,
        height=HEIGHT,
        fps=30,
        total_frames=frames,
        palette=PALETTE,
        seed=seed,
        rng=random.Random(seed),
    )


def test_loop_position_returns_to_its_start():
    assert loop_position(0.0) == pytest.approx(0.0)
    assert loop_position(0.5) == pytest.approx(1.0)
    assert loop_position(1.0) == pytest.approx(0.0, abs=1e-9)


def test_ease_is_smooth_and_bounded():
    assert ease(0.0) == 0.0
    assert ease(1.0) == 1.0
    assert 0.0 < ease(0.5) < 1.0


def test_load_image_covers_the_frame(tmp_path: Path):
    image = synthetic(tmp_path, "a.png", (200, 60, 160))
    # Oversampled so a zoom has material to work with.
    assert image.shape[0] >= HEIGHT
    assert image.shape[1] >= WIDTH
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


def test_frames_have_the_right_shape(tmp_path: Path):
    animator = StillAnimator([synthetic(tmp_path, "a.png", (200, 60, 160))])
    ctx = context()
    animator.prepare(ctx)
    frame = animator.render(0, ctx)
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8


def test_the_camera_actually_moves(tmp_path: Path):
    animator = StillAnimator([synthetic(tmp_path, "a.png", (200, 60, 160))])
    ctx = context()
    animator.prepare(ctx)
    first = animator.render(0, ctx).astype(np.int16)
    middle = animator.render(20, ctx).astype(np.int16)
    assert np.abs(first - middle).mean() > 1.0


def test_the_clip_loops_exactly(tmp_path: Path):
    """Frame N must be frame 0 again; that is what loops_natively promises."""
    animator = StillAnimator([synthetic(tmp_path, "a.png", (200, 60, 160))])
    ctx = context(frames=40)
    animator.prepare(ctx)
    assert np.array_equal(animator.render(0, ctx), animator.render(40, ctx))


def test_several_images_are_morphed_together(tmp_path: Path):
    images = [
        synthetic(tmp_path, "a.png", (255, 0, 0)),
        synthetic(tmp_path, "b.png", (0, 255, 0)),
        synthetic(tmp_path, "c.png", (0, 0, 255)),
    ]
    animator = StillAnimator(images)
    ctx = context(frames=60)
    animator.prepare(ctx)
    # The dominant channel should shift across the clip as it dissolves.
    early = animator.render(2, ctx).mean(axis=(0, 1))
    middle = animator.render(30, ctx).mean(axis=(0, 1))
    assert np.argmax(early) != np.argmax(middle)


def test_no_images_is_refused():
    with pytest.raises(ValueError, match="at least one image"):
        StillAnimator([])


def test_describe_records_the_prompts(tmp_path: Path):
    animator = StillAnimator(
        [synthetic(tmp_path, "a.png", (200, 60, 160))],
        metadata=[{"prompt": "neon tunnel", "seed": 42}],
    )
    ctx = context()
    animator.prepare(ctx)
    described = animator.describe(ctx)
    assert described["prompts"] == ["neon tunnel"]
    assert described["seeds"] == [42]


# --- prompt presets ------------------------------------------------------


def test_the_shipped_presets_load():
    path = Path(__file__).resolve().parent.parent / "config" / "prompts.json"
    prompts = PromptSet.load(path)
    assert len(prompts) >= 12
    assert prompts.by_id("deep_space_nebula") is not None


def test_the_negative_prompt_excludes_faces_and_text():
    path = Path(__file__).resolve().parent.parent / "config" / "prompts.json"
    negative = PromptSet.load(path).negative_prompt
    for banned in ("text", "watermark", "logo", "face"):
        assert banned in negative


def test_a_missing_prompt_file_falls_back(tmp_path: Path):
    prompts = PromptSet.load(tmp_path / "nope.json")
    assert len(prompts) == 1
    assert prompts.negative_prompt == DEFAULT_NEGATIVE


def test_disabled_presets_are_skipped(tmp_path: Path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "presets": [
                    {"id": "on", "prompt": "a"},
                    {"id": "off", "prompt": "b", "enabled": False},
                    {"id": "empty", "prompt": "   "},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [p.id for p in PromptSet.load(path)] == ["on"]


def test_the_style_suffix_is_appended(tmp_path: Path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps({"style_suffix": "abstract", "presets": [{"id": "a", "prompt": "neon"}]}),
        encoding="utf-8",
    )
    prompts = PromptSet.load(path)
    assert prompts.all[0].full_prompt(prompts.style_suffix) == "neon, abstract"


# --- backend selection ---------------------------------------------------


def test_every_backend_probes_without_raising(tmp_path: Path):
    """Probe runs on every dashboard refresh; it must never throw."""
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path)})
    paths = Paths.from_config(cfg)
    infos = ai_backends.probe_all(cfg, paths.models)
    assert {i.name for i in infos} == {"cuda", "openvino", "sdcpp"}
    for info in infos:
        assert info.reason, f"{info.name} gave no reason"


def test_none_is_respected(tmp_path: Path):
    cfg = Config.model_validate({"visual": {"ai": {"backend": "none"}}})
    backend, reason = ai_backends.resolve(cfg, tmp_path)
    assert backend is None
    assert "disabled" in reason


def test_auto_explains_itself_when_nothing_is_available(tmp_path: Path):
    cfg = Config()
    backend, reason = ai_backends.resolve(cfg, tmp_path)
    assert backend is None
    for name in ("cuda", "sdcpp", "openvino"):
        assert name in reason


def test_asking_for_an_unavailable_backend_degrades(tmp_path: Path):
    """A config written for a GPU host must still boot on a CPU-only VM."""
    cfg = Config.model_validate({"visual": {"ai": {"backend": "cuda"}}})
    backend, reason = ai_backends.resolve(cfg, tmp_path)
    assert backend is None
    assert "cuda" in reason
