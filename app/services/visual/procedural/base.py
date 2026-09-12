"""The procedural generator plugin interface.

A generator is a pure function of (frame index, context) to an RGB frame. It
holds no files, spawns no processes and knows nothing about encoding — the
animator handles all of that — which keeps each one small enough to read in
one sitting and testable without ffmpeg.

Adding one: subclass, decorate with ``@register``, import it in
``__init__.py``. Its name then works in ``visual.procedural.generators``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.services.visual.palettes import Palette


@dataclass
class RenderContext:
    """Everything a generator is allowed to depend on."""

    width: int
    height: int
    fps: int
    total_frames: int
    palette: Palette
    seed: int
    rng: random.Random = field(repr=False, default_factory=random.Random)
    # Cached so the LUT is built once per clip rather than once per frame.
    lut: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lut is None:
            self.lut = self.palette.lut()

    @property
    def duration_s(self) -> float:
        return self.total_frames / max(1, self.fps)

    def phase(self, frame: int) -> float:
        """Position through the clip in [0, 1)."""
        return (frame % max(1, self.total_frames)) / max(1, self.total_frames)


class ProceduralGenerator(ABC):
    """Base class for every procedural visual."""

    #: Identifier used in configuration and stored in block metadata.
    name: str = "unnamed"
    #: Shown in the dashboard.
    description: str = ""
    #: True when the generator's own maths returns to its starting state at
    #: the end of the clip. Those that evolve irreversibly (reaction-diffusion,
    #: particle trails) say False and the animator blends the ends instead.
    loops_natively: bool = True

    def prepare(self, ctx: RenderContext) -> None:  # noqa: B027
        """Allocate buffers and pick per-clip random parameters.

        Optional on purpose: a stateless generator has nothing to set up,
        so this is a hook rather than an abstract method.
        """

    @abstractmethod
    def render(self, frame: int, ctx: RenderContext) -> np.ndarray:
        """Return one frame as a (height, width, 3) uint8 array."""

    def describe(self, ctx: RenderContext) -> dict[str, object]:
        """Provenance recorded in the block sidecar."""
        return {
            "generator": self.name,
            "palette": ctx.palette.id,
            "seed": ctx.seed,
            **{k: v for k, v in ctx.params.items() if isinstance(v, (int, float, str, bool))},
        }


_REGISTRY: dict[str, type[ProceduralGenerator]] = {}


def register(cls: type[ProceduralGenerator]) -> type[ProceduralGenerator]:
    if not cls.name or cls.name == "unnamed":
        raise ValueError(f"{cls.__name__} must set a name")
    if cls.name in _REGISTRY:
        raise ValueError(f"two generators are both called {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> dict[str, type[ProceduralGenerator]]:
    return dict(_REGISTRY)


def get(name: str) -> type[ProceduralGenerator]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown generator {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def to_uint8(values: np.ndarray) -> np.ndarray:
    """Clamp a float image in [0, 1] to uint8 without wrapping.

    Casting straight to uint8 wraps 1.02 round to 5, which shows up as bright
    speckle exactly where the image is brightest.
    """
    return (np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def upsample(field: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinearly resize a 2D float field.

    Several generators produce inherently smooth output — warped noise, a
    chemical concentration — and computing it at full resolution is wasted
    work: the result is upscaled again by ffmpeg's lanczos pass anyway.
    Rendering at half size and interpolating here is a 4x saving that costs
    nothing visible.
    """
    source_h, source_w = field.shape[:2]
    if (source_h, source_w) == (height, width):
        return field

    ys = np.linspace(0.0, source_h - 1, height, dtype=np.float32)
    xs = np.linspace(0.0, source_w - 1, width, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, source_h - 1)
    x1 = np.minimum(x0 + 1, source_w - 1)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]

    top_left = field[np.ix_(y0, x0)]
    top_right = field[np.ix_(y0, x1)]
    bottom_left = field[np.ix_(y1, x0)]
    bottom_right = field[np.ix_(y1, x1)]

    top = top_left + (top_right - top_left) * tx
    bottom = bottom_left + (bottom_right - bottom_left) * tx
    return (top + (bottom - top) * ty).astype(np.float32)


# Below this spread a field is flat and rescaling it only amplifies noise.
FLAT_FIELD_EPSILON = 1e-9


def normalise(values: np.ndarray) -> np.ndarray:
    """Rescale an arbitrary field into [0, 1] using its own range."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    low = float(values[finite].min())
    high = float(values[finite].max())
    if high - low < FLAT_FIELD_EPSILON:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


__all__ = [
    "ProceduralGenerator",
    "RenderContext",
    "available",
    "get",
    "normalise",
    "register",
    "to_uint8",
    "upsample",
]
