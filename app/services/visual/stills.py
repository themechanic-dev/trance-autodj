"""Turn still images into moving clips.

An AI still is a photograph of nothing; on its own it would be a freeze-frame
for thirty seconds. This gives it motion using the same interface as the
procedural generators, so the animator and block builder need no special case:
a clip is a clip.

Three techniques, combined per clip:

* **Ken Burns** — a slow eased zoom and pan. Eased, not linear, because a
  linear zoom reads as a machine moving and an eased one reads as a camera.
* **Displacement** — the sampling grid is warped by slowly drifting noise,
  which makes a flat image behave like something viscous.
* **Morph** — with more than one image, the clip dissolves between them, so
  the picture evolves rather than merely moving.

It loops natively: every path is driven by a cosine of the clip phase, so the
last frame is the first frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.services.visual.procedural.base import ProceduralGenerator, RenderContext
from app.services.visual.procedural.noise import value_noise_3d

log = get_logger(__name__)

# How far in the source image the crop may travel, as a fraction of its size.
MAX_PAN = 0.18
# Zoom range. 1.0 means "the whole image"; smaller crops in further.
ZOOM_NEAR, ZOOM_FAR = 0.62, 0.92
# Displacement strength in pixels at the render resolution.
DISPLACE_MIN, DISPLACE_MAX = 2.0, 9.0


def load_image(path: Path, width: int, height: int) -> np.ndarray:
    """Load and letterbox-free-fit an image to at least (height, width).

    The image is scaled to *cover* the frame so the Ken Burns crop always has
    material to work with, and converted to RGB so a greyscale or RGBA still
    does not break the frame contract downstream.
    """
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        source_ratio = image.width / image.height
        target_ratio = width / height
        # Oversample so that zooming in does not reveal interpolation.
        if source_ratio > target_ratio:
            new_height = int(height / ZOOM_NEAR)
            new_width = int(new_height * source_ratio)
        else:
            new_width = int(width / ZOOM_NEAR)
            new_height = int(new_width / source_ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def ease(t: float) -> float:
    """Smoothstep. A linear pan looks mechanical; this looks like a camera."""
    return t * t * (3.0 - 2.0 * t)


def loop_position(phase: float) -> float:
    """0 -> 1 -> 0 across the clip, so the motion returns where it began."""
    return (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0


@dataclass
class _Path:
    """Where the crop starts and ends."""

    start: tuple[float, float, float]  # x, y (0..1 of slack), zoom
    end: tuple[float, float, float]


class StillAnimator(ProceduralGenerator):
    """Animates one or more stills. Constructed with images, not by name."""

    name = "stills"
    description = "AI stills brought to life with a slow camera and liquid warp"
    loops_natively = True

    def __init__(self, images: list[np.ndarray], *, metadata: list[dict] | None = None) -> None:
        if not images:
            raise ValueError("the still animator needs at least one image")
        self._images = images
        self._metadata = metadata or []
        self._paths: list[_Path] = []
        self._grid: tuple[np.ndarray, np.ndarray] | None = None

    def prepare(self, ctx: RenderContext) -> None:
        rng = ctx.rng
        for _ in self._images:
            self._paths.append(
                _Path(
                    start=(rng.random(), rng.random(), rng.uniform(ZOOM_FAR, 1.0)),
                    end=(rng.random(), rng.random(), rng.uniform(ZOOM_NEAR, ZOOM_FAR)),
                )
            )

        ys, xs = np.mgrid[0 : ctx.height, 0 : ctx.width]
        self._grid = (xs.astype(np.float32), ys.astype(np.float32))

        ctx.params.update(
            {
                "images": len(self._images),
                "displace": rng.uniform(DISPLACE_MIN, DISPLACE_MAX),
                "displace_scale": rng.uniform(0.004, 0.012),
                "sources": [m.get("preset", "") for m in self._metadata if m],
            }
        )

    # -- sampling ----------------------------------------------------------

    def _crop(self, image: np.ndarray, path: _Path, t: float, ctx: RenderContext) -> np.ndarray:
        source_h, source_w = image.shape[:2]
        eased = ease(t)

        zoom = path.start[2] + (path.end[2] - path.start[2]) * eased
        crop_w = min(source_w, int(source_w * zoom))
        crop_h = min(source_h, int(source_h * zoom))

        slack_x = max(0, source_w - crop_w)
        slack_y = max(0, source_h - crop_h)
        pan_x = path.start[0] + (path.end[0] - path.start[0]) * eased
        pan_y = path.start[1] + (path.end[1] - path.start[1]) * eased
        left = int(slack_x * min(1.0, MAX_PAN + pan_x * (1.0 - MAX_PAN)))
        top = int(slack_y * min(1.0, MAX_PAN + pan_y * (1.0 - MAX_PAN)))

        left = max(0, min(left, source_w - crop_w))
        top = max(0, min(top, source_h - crop_h))

        window = image[top : top + crop_h, left : left + crop_w]
        return self._resize(window, ctx.height, ctx.width)

    @staticmethod
    def _resize(window: np.ndarray, height: int, width: int) -> np.ndarray:
        """Nearest-neighbour resample. Cheap, and ffmpeg's lanczos follows."""
        source_h, source_w = window.shape[:2]
        ys = (np.arange(height) * source_h // height).clip(0, source_h - 1)
        xs = (np.arange(width) * source_w // width).clip(0, source_w - 1)
        return window[np.ix_(ys, xs)]

    def _displace(self, frame: np.ndarray, phase: float, ctx: RenderContext) -> np.ndarray:
        strength = float(ctx.params["displace"])
        if strength <= 0 or self._grid is None:
            return frame
        xs, ys = self._grid
        scale = np.float32(ctx.params["displace_scale"])
        z = phase * 4.0  # one full turn of the noise lattice per clip

        dx = value_noise_3d(xs * scale, ys * scale, z, period=(8, 8, 4), seed=ctx.seed) - 0.5
        dy = (
            value_noise_3d(
                xs * scale + 31.7, ys * scale + 11.3, z, period=(8, 8, 4), seed=ctx.seed + 977
            )
            - 0.5
        )

        sample_x = np.clip(xs + dx * (2.0 * strength), 0, ctx.width - 1).astype(np.int32)
        sample_y = np.clip(ys + dy * (2.0 * strength), 0, ctx.height - 1).astype(np.int32)
        return frame[sample_y, sample_x]

    def render(self, frame: int, ctx: RenderContext) -> np.ndarray:
        phase = ctx.phase(frame)
        position = loop_position(phase)

        if len(self._images) == 1:
            result = self._crop(self._images[0], self._paths[0], position, ctx)
        else:
            # Walk through the images across the clip, dissolving between
            # consecutive pairs.
            span = 1.0 / (len(self._images) - 1) if len(self._images) > 1 else 1.0
            index = min(int(position / span), len(self._images) - 2)
            local = (position - index * span) / span

            first = self._crop(self._images[index], self._paths[index], position, ctx)
            second = self._crop(self._images[index + 1], self._paths[index + 1], position, ctx)
            blend = ease(local)
            result = (
                first.astype(np.float32) * (1.0 - blend) + second.astype(np.float32) * blend
            ).astype(np.uint8)

        return self._displace(result, phase, ctx)

    def describe(self, ctx: RenderContext) -> dict[str, object]:
        return {
            "generator": "stills",
            "palette": ctx.palette.id,
            "seed": ctx.seed,
            "images": len(self._images),
            "prompts": [m.get("prompt", "") for m in self._metadata if m],
            "seeds": [m.get("seed") for m in self._metadata if m],
        }


__all__ = ["StillAnimator", "ease", "load_image", "loop_position"]
