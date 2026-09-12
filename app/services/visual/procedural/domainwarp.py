"""fBm domain warping — feed noise its own output as coordinates.

The technique is Inigo Quilez's: warp the sampling position by one noise
field, then warp that by another. It produces the drifting, marbled forms
that read as "liquid" rather than "static texture".

Time is the noise field's third dimension, sampled over exactly one lattice
period, so the clip loops without any crossfade.
"""

from __future__ import annotations

import numpy as np

from app.services.visual.procedural.base import (
    ProceduralGenerator,
    RenderContext,
    register,
    upsample,
)
from app.services.visual.procedural.noise import fbm3, pixel_grid


@register
class DomainWarp(ProceduralGenerator):
    name = "domainwarp"
    description = "Marbled fields from noise warped by noise"
    loops_natively = True

    #: Rendered at half size and interpolated back up. Warped fBm has no
    #: detail at the pixel level to lose, and this is a 4x saving on the most
    #: expensive generator in the set.
    INTERNAL_SCALE = 0.5

    def prepare(self, ctx: RenderContext) -> None:
        rng = ctx.rng
        self._w = max(64, int(ctx.width * self.INTERNAL_SCALE))
        self._h = max(64, int(ctx.height * self.INTERNAL_SCALE))
        x, y = pixel_grid(self._w, self._h)
        scale = rng.uniform(1.1, 2.2)
        self._x = (x * scale).astype(np.float32)
        self._y = (y * scale).astype(np.float32)
        # The z period is the loop length in noise-lattice units; the clip
        # walks exactly once around it.
        self._z_period = 4
        ctx.params.update(
            {
                # Three octaves, not five. Beyond three the extra detail is
                # below what survives the upscale and the encoder anyway.
                "octaves": rng.choice([2, 3, 3]),
                "warp": rng.uniform(1.8, 3.8),
                "gain": rng.uniform(0.45, 0.58),
                "contrast": rng.uniform(0.9, 1.5),
                "internal_scale": self.INTERNAL_SCALE,
            }
        )

    def _fbm(self, x: np.ndarray, y: np.ndarray, z: float, seed_offset: int, ctx) -> np.ndarray:
        return fbm3(
            x,
            y,
            z,
            octaves=int(ctx.params["octaves"]),
            gain=float(ctx.params["gain"]),
            period=(8, 8, self._z_period),
            seed=ctx.seed + seed_offset,
        )

    def render(self, frame: int, ctx: RenderContext) -> np.ndarray:
        z = ctx.phase(frame) * self._z_period
        x, y = self._x, self._y
        warp = np.float32(ctx.params["warp"])

        # One warp level, three fBm evaluations. The textbook version warps
        # twice (five evaluations) but at this scale the second pass is not
        # distinguishable from a slightly stronger first one, and it cost
        # two thirds more time per frame.
        qx = self._fbm(x, y, z, 0, ctx)
        qy = self._fbm(x + np.float32(5.2), y + np.float32(1.3), z, 101, ctx)
        field = self._fbm(x + warp * qx, y + warp * qy, z, 401, ctx)

        contrast = np.float32(ctx.params["contrast"])
        shaped = np.clip((field - 0.5) * contrast + 0.5, 0.0, 1.0)
        return ctx.palette.map(upsample(shaped, ctx.height, ctx.width), ctx.lut)
