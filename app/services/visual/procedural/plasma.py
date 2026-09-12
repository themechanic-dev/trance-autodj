"""Classic sine plasma — the oldest trick in the demoscene, still beautiful.

Loops exactly: every term's time argument advances by a whole multiple of 2π
across the clip, so the last frame is the first frame.
"""

from __future__ import annotations

import numpy as np

from app.services.visual.procedural.base import (
    ProceduralGenerator,
    RenderContext,
    register,
)
from app.services.visual.procedural.noise import pixel_grid


@register
class Plasma(ProceduralGenerator):
    name = "plasma"
    description = "Interfering sine fields in slow drift"
    loops_natively = True

    def prepare(self, ctx: RenderContext) -> None:
        rng = ctx.rng
        self._x, self._y = pixel_grid(ctx.width, ctx.height)
        self._r = np.sqrt(self._x**2 + self._y**2).astype(np.float32)
        # Spatial frequencies. Kept low: trance visuals want slow, wide forms,
        # not a busy texture that flickers once it is scaled up to 720p.
        ctx.params.update(
            {
                "fx": rng.uniform(1.6, 3.4),
                "fy": rng.uniform(1.6, 3.4),
                "fd": rng.uniform(1.2, 2.8),
                "fr": rng.uniform(2.0, 4.5),
                # Integer time multipliers are what make the loop exact.
                "kx": rng.choice([1, 1, 2]),
                "ky": rng.choice([1, 2, 2]),
                "kd": rng.choice([1, 2, 3]),
                "kr": rng.choice([1, 2]),
                "swirl": rng.uniform(0.0, 0.35),
            }
        )

    def render(self, frame: int, ctx: RenderContext) -> np.ndarray:
        p = ctx.params
        t = 2.0 * np.pi * ctx.phase(frame)
        x, y, r = self._x, self._y, self._r

        if p["swirl"]:
            angle = np.float32(p["swirl"]) * np.sin(t) * r
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x, y = x * cos_a - y * sin_a, x * sin_a + y * cos_a

        v = np.sin(x * p["fx"] + t * p["kx"], dtype=np.float32)
        v += np.sin(y * p["fy"] + t * p["ky"], dtype=np.float32)
        v += np.sin((x + y) * p["fd"] + t * p["kd"], dtype=np.float32)
        v += np.sin(r * p["fr"] - t * p["kr"], dtype=np.float32)

        return ctx.palette.map((v + 4.0) / 8.0, ctx.lut)
