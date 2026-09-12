"""Every generator must produce a correct frame, for every seed, always."""

from __future__ import annotations

import random

import numpy as np
import pytest

from app.services.visual.palettes import Palette
from app.services.visual.procedural import available, get
from app.services.visual.procedural.base import RenderContext, normalise, upsample
from app.services.visual.procedural.noise import fbm3, value_noise_3d

WIDTH, HEIGHT, FPS = 96, 64, 30

PALETTE = Palette(
    id="test",
    name="Test",
    colors=("#000000", "#0b4f8a", "#5ee7f0", "#ffffff"),
)

ALL_GENERATORS = sorted(available())


def make_context(seed: int = 1, frames: int = 20) -> RenderContext:
    return RenderContext(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        total_frames=frames,
        palette=PALETTE,
        seed=seed,
        rng=random.Random(seed),
    )


def test_the_expected_six_generators_are_registered():
    assert ALL_GENERATORS == [
        "domainwarp",
        "flowfield",
        "plasma",
        "reaction_diffusion",
        "tunnel",
        "waves",
    ]


@pytest.mark.parametrize("name", ALL_GENERATORS)
def test_frame_shape_and_dtype(name: str):
    generator = get(name)()
    ctx = make_context()
    generator.prepare(ctx)
    frame = generator.render(0, ctx)
    assert frame.shape == (HEIGHT, WIDTH, 3), name
    assert frame.dtype == np.uint8, name


@pytest.mark.parametrize("name", ALL_GENERATORS)
def test_the_picture_actually_moves(name: str):
    """A frozen generator looks like a working one in a still screenshot.

    Reaction-diffusion shipped exactly that way once: an unstable Laplacian
    diverged, the clamp froze it, and every frame was identical.
    """
    generator = get(name)()
    ctx = make_context(frames=40)
    generator.prepare(ctx)
    first = generator.render(0, ctx)
    later = [generator.render(i, ctx) for i in range(1, 6)][-1]
    assert not np.array_equal(first, later), f"{name} produced a still image"


@pytest.mark.parametrize("name", ALL_GENERATORS)
def test_output_uses_a_real_range_of_the_palette(name: str):
    """A frame that is all one colour means the field collapsed."""
    generator = get(name)()
    ctx = make_context()
    generator.prepare(ctx)
    frame = generator.render(3, ctx)
    assert frame.std() > 2.0, f"{name} is nearly flat (std={frame.std():.2f})"


@pytest.mark.parametrize("name", ALL_GENERATORS)
def test_same_seed_reproduces_the_same_frames(name: str):
    """The sidecar records a seed; it has to be enough to rebuild the clip."""
    frames = []
    for _ in range(2):
        generator = get(name)()
        ctx = make_context(seed=4242)
        generator.prepare(ctx)
        frames.append([generator.render(i, ctx) for i in range(3)])
    for a, b in zip(frames[0], frames[1], strict=True):
        assert np.array_equal(a, b), f"{name} is not reproducible from its seed"


@pytest.mark.parametrize("name", ALL_GENERATORS)
def test_many_seeds_and_frames_never_raise(name: str):
    """Regression: flowfield crashed with IndexError on some seeds only.

    A float32 coordinate of -1e-8 taken modulo the height rounds to exactly
    the height, which then indexes one past the end of the trail buffer. It
    only happened for some seeds, and it killed two of three blocks in a row.
    """
    for seed in range(6):
        generator = get(name)()
        ctx = make_context(seed=seed, frames=25)
        generator.prepare(ctx)
        for frame in range(25):
            image = generator.render(frame, ctx)
            assert np.isfinite(image).all()


@pytest.mark.parametrize("name", ALL_GENERATORS)
def test_generators_declare_whether_they_loop(name: str):
    cls = get(name)
    assert isinstance(cls.loops_natively, bool)
    assert cls.description, f"{name} has no description for the UI"


def test_natively_looping_generators_really_do_loop():
    """Frame N must be frame 0 again, or the clip visibly jumps when reused."""
    for name in ALL_GENERATORS:
        cls = get(name)
        if not cls.loops_natively:
            continue
        generator = cls()
        ctx = make_context(seed=11, frames=24)
        generator.prepare(ctx)
        first = generator.render(0, ctx).astype(np.int16)
        wrapped = generator.render(24, ctx).astype(np.int16)  # phase() wraps
        difference = np.abs(first - wrapped).mean()
        assert difference < 1.0, f"{name} does not close its loop (mean diff {difference:.2f})"


# --- noise ---------------------------------------------------------------


def test_value_noise_is_in_range():
    x, y = np.meshgrid(
        np.linspace(0, 8, 40, dtype=np.float32), np.linspace(0, 8, 40, dtype=np.float32)
    )
    values = value_noise_3d(x, y, 0.5, period=(8, 8, 8), seed=3)
    assert values.min() >= 0.0
    assert values.max() < 1.0


def test_noise_tiles_on_its_period():
    """Tileability is what gives four of the generators a free perfect loop."""
    x, y = np.meshgrid(
        np.linspace(0, 4, 24, dtype=np.float32), np.linspace(0, 4, 24, dtype=np.float32)
    )
    period = (8, 8, 6)
    at_zero = value_noise_3d(x, y, 0.0, period=period, seed=9)
    at_period = value_noise_3d(x, y, float(period[2]), period=period, seed=9)
    assert np.allclose(at_zero, at_period, atol=1e-5)


def test_fbm_is_normalised_and_finite():
    x, y = np.meshgrid(
        np.linspace(0, 6, 32, dtype=np.float32), np.linspace(0, 6, 32, dtype=np.float32)
    )
    values = fbm3(x, y, 1.0, octaves=4, period=(8, 8, 8), seed=2)
    assert np.isfinite(values).all()
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_noise_differs_between_seeds():
    x, y = np.meshgrid(
        np.linspace(0, 5, 20, dtype=np.float32), np.linspace(0, 5, 20, dtype=np.float32)
    )
    a = value_noise_3d(x, y, 0.0, period=(8, 8, 8), seed=1)
    b = value_noise_3d(x, y, 0.0, period=(8, 8, 8), seed=2)
    assert not np.allclose(a, b)


# --- helpers -------------------------------------------------------------


def test_upsample_preserves_corners_and_size():
    field = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    bigger = upsample(field, 8, 8)
    assert bigger.shape == (8, 8)
    assert bigger[0, 0] == pytest.approx(0.0)
    assert bigger[0, -1] == pytest.approx(1.0)
    assert bigger[-1, 0] == pytest.approx(1.0)


def test_upsample_is_a_noop_at_the_same_size():
    field = np.zeros((4, 4), dtype=np.float32)
    assert upsample(field, 4, 4) is field


def test_normalise_handles_a_flat_field():
    assert normalise(np.full((4, 4), 7.0, dtype=np.float32)).max() == 0.0


def test_normalise_ignores_non_finite_values():
    field = np.array([[0.0, 1.0], [np.nan, np.inf]], dtype=np.float32)
    result = normalise(field)
    assert np.isfinite(result[0]).all()
