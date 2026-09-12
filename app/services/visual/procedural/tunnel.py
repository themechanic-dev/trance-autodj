"""Infinite zoom tunnel.

Polar coordinates turn a flat texture into a corridor: the angle becomes one
texture axis and 1/radius becomes the other, so a constant scroll along that
axis reads as endless forward motion.

It loops because the texture repeats along its depth axis and the clip
scrolls a whole number of repeats.
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
class Tunnel(ProceduralGenerator):
    name = "tunnel"
    description = "Endless zoom down a textured corridor"
    loops_natively = True

    #: Smooth radial content; half resolution is invisible after the
    #: lanczos upscale and halves the trig and noise work.
    INTERNAL_SCALE = 0.5

    def prepare(self, ctx: RenderContext) -> None:
        rng = ctx.rng
        self._w = max(64, int(ctx.width * self.INTERNAL_SCALE))
        self._h = max(64, int(ctx.height * self.INTERNAL_SCALE))
        x, y = pixel_grid(self._w, self._h)
        # Keep the singularity at r = 0 off the pixel grid.
        radius = np.sqrt(x**2 + y**2).astype(np.float32) + np.float32(1e-3)
        self._angle = np.arctan2(y, x).astype(np.float32)
        self._depth = (np.float32(1.0) / radius).astype(np.float32)
        # Darken the mouth of the tunnel so the centre reads as distance
        # rather than as a bright artefact.
        self._vignette = np.clip(radius * np.float32(1.15), 0.0, 1.0) ** np.float32(0.85)
        ctx.params.update(
            {
                "repeats": rng.choice([2, 3, 3, 4]),
                "twist": rng.uniform(-1.2, 1.2),
                "ribs": rng.choice([6, 8, 10, 12, 16]),
                "octaves": rng.choice([2, 3, 3]),
                "noise_mix": rng.uniform(0.35, 0.8),
                "internal_scale": self.INTERNAL_SCALE,
            }
        )

    def render(self, frame: int, ctx: RenderContext) -> np.ndarray:
        p = ctx.params
        phase = ctx.phase(frame)
        repeats = int(p["repeats"])

        depth = self._depth * repeats + phase * repeats
        angle = self._angle + np.float32(p["twist"]) * phase * 2.0 * np.pi

        ribs = 0.5 + 0.5 * np.sin(angle * int(p["ribs"]) + depth * 2.0, dtype=np.float32)
        rings = 0.5 + 0.5 * np.sin(depth * 2.0 * np.pi, dtype=np.float32)

        texture = fbm3(
            np.cos(angle) * np.float32(2.0),
            np.sin(angle) * np.float32(2.0),
            depth,
            octaves=int(p["octaves"]),
            period=(8, 8, repeats),
            seed=ctx.seed,
        )

        mix = np.float32(p["noise_mix"])
        field = (1.0 - mix) * (0.6 * ribs + 0.4 * rings) + mix * texture
        shaped = np.clip(field * self._vignette, 0.0, 1.0)
        return ctx.palette.map(upsample(shaped, ctx.height, ctx.width), ctx.lut)
