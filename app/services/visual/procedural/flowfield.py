"""Particles following a curl-noise vector field, leaving trails.

The trail buffer is history, so this cannot loop by construction — it says so
and lets the animator blend the ends instead.
"""

from __future__ import annotations

import numpy as np

from app.services.visual.procedural.base import (
    ProceduralGenerator,
    RenderContext,
    register,
)
from app.services.visual.procedural.noise import value_noise_3d

# Particles per megapixel of render area, so the density looks the same at
# every resolution instead of thinning out as the frame grows.
PARTICLES_PER_MEGAPIXEL = 26000


@register
class FlowField(ProceduralGenerator):
    name = "flowfield"
    description = "Particle trails drifting through a curl-noise field"
    loops_natively = False

    def prepare(self, ctx: RenderContext) -> None:
        rng = np.random.default_rng(ctx.seed)
        megapixels = (ctx.width * ctx.height) / 1_000_000
        count = max(2000, int(PARTICLES_PER_MEGAPIXEL * megapixels))

        self._px = rng.random(count, dtype=np.float32) * ctx.width
        self._py = rng.random(count, dtype=np.float32) * ctx.height
        self._age = rng.integers(0, 180, size=count).astype(np.int32)
        self._rng = rng
        self._trails = np.zeros((ctx.height, ctx.width), dtype=np.float32)

        ctx.params.update(
            {
                "particles": count,
                "decay": ctx.rng.uniform(0.955, 0.985),
                "speed": ctx.rng.uniform(0.9, 2.1),
                "noise_scale": ctx.rng.uniform(0.004, 0.011),
                "field_drift": ctx.rng.uniform(0.15, 0.5),
                "lifetime": ctx.rng.randint(120, 300),
                "glow": ctx.rng.uniform(0.6, 1.0),
            }
        )
        self._z_period = 8

    def _velocity(self, z: float, ctx: RenderContext) -> tuple[np.ndarray, np.ndarray]:
        scale = np.float32(ctx.params["noise_scale"])
        x = self._px * scale
        y = self._py * scale
        # Two decorrelated fields make an angle; cheaper than a true curl and
        # visually indistinguishable at these speeds.
        angle = value_noise_3d(
            x, y, z, period=(16, 16, self._z_period), seed=ctx.seed
        ) * np.float32(4.0 * np.pi)
        speed = np.float32(ctx.params["speed"])
        return np.cos(angle) * speed, np.sin(angle) * speed

    def render(self, frame: int, ctx: RenderContext) -> np.ndarray:
        p = ctx.params
        z = ctx.phase(frame) * self._z_period * float(p["field_drift"])

        self._trails *= np.float32(p["decay"])

        vx, vy = self._velocity(z, ctx)
        self._px += vx
        self._py += vy
        self._age += 1

        # Wrap rather than clamp: clamping piles every particle onto the
        # borders within a few seconds and the frame grows a bright frame.
        np.mod(self._px, ctx.width, out=self._px)
        np.mod(self._py, ctx.height, out=self._py)

        # Respawn old particles so the field keeps renewing instead of
        # settling into the field's attractors and going still.
        expired = self._age > int(p["lifetime"])
        if expired.any():
            n = int(expired.sum())
            self._px[expired] = self._rng.random(n, dtype=np.float32) * ctx.width
            self._py[expired] = self._rng.random(n, dtype=np.float32) * ctx.height
            self._age[expired] = 0

        # np.mod on float32 can return exactly the modulus for a tiny
        # negative input (-1e-8 % 360 rounds to 360.0), which then indexes one
        # past the end of the buffer. Clipping the integer indices is the only
        # reliable guard; checking the floats beforehand is not, because the
        # rounding happens in the cast itself.
        xi = np.clip(self._px.astype(np.int32), 0, ctx.width - 1)
        yi = np.clip(self._py.astype(np.int32), 0, ctx.height - 1)
        np.add.at(self._trails, (yi, xi), np.float32(p["glow"]))

        field = np.tanh(self._trails * np.float32(0.55))
        return ctx.palette.map(field, ctx.lut)
