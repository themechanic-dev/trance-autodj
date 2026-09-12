"""Interference and moire from summed plane waves.

Each wave's phase advances by a whole number of turns across the clip, so the
sum returns to itself exactly.
"""

from __future__ import annotations

import numpy as np

from app.services.visual.procedural.base import (
    ProceduralGenerator,
    RenderContext,
    register,
)
from app.services.visual.procedural.noise import pixel_grid

# How often a radial term joins the plane waves.
RADIAL_CHANCE = 0.4


@register
class Waves(ProceduralGenerator):
    name = "waves"
    description = "Moire interference between drifting plane waves"
    loops_natively = True

    def prepare(self, ctx: RenderContext) -> None:
        rng = ctx.rng
        self._x, self._y = pixel_grid(ctx.width, ctx.height)
        count = rng.randint(3, 5)
        self._waves = [
            {
                "angle": rng.uniform(0, np.pi),
                "freq": rng.uniform(4.0, 13.0),
                "turns": rng.choice([1, 1, 2, 2, 3]),
                "amp": rng.uniform(0.6, 1.0),
                "curve": rng.uniform(0.0, 0.6),
            }
            for _ in range(count)
        ]
        ctx.params.update(
            {
                "waves": count,
                "sharpen": rng.uniform(1.0, 2.6),
                "radial": rng.random() < RADIAL_CHANCE,
            }
        )
        if ctx.params["radial"]:
            self._r = np.sqrt(self._x**2 + self._y**2).astype(np.float32)

    def render(self, frame: int, ctx: RenderContext) -> np.ndarray:
        phase = ctx.phase(frame)
        total = np.zeros_like(self._x, dtype=np.float32)
        amplitude_sum = 0.0

        for wave in self._waves:
            direction = self._x * np.cos(wave["angle"]) + self._y * np.sin(wave["angle"])
            if wave["curve"]:
                direction = direction + wave["curve"] * (self._x**2 - self._y**2)
            total += wave["amp"] * np.sin(
                direction * wave["freq"] + 2.0 * np.pi * wave["turns"] * phase,
                dtype=np.float32,
            )
            amplitude_sum += wave["amp"]

        if ctx.params["radial"]:
            total += np.sin(self._r * 9.0 - 2.0 * np.pi * phase, dtype=np.float32)
            amplitude_sum += 1.0

        field = total / max(amplitude_sum, 1e-6)
        # tanh keeps the banding crisp without clipping into flat plateaus.
        shaped = np.tanh(field * np.float32(ctx.params["sharpen"])) * 0.5 + 0.5
        return ctx.palette.map(shaped, ctx.lut)
