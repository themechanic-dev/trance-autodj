"""Gray-Scott reaction-diffusion: slow, organic, faintly alive.

Two chemicals diffuse at different rates while one converts the other. The
parameter pair (feed, kill) decides whether you get coral, mitosis or worms;
the presets below are the ones that stay interesting for minutes rather than
settling within seconds.

It evolves irreversibly, so it does not loop; the animator blends the ends.
"""

from __future__ import annotations

import numpy as np

from app.services.visual.procedural.base import (
    ProceduralGenerator,
    RenderContext,
    normalise,
    register,
)

# (feed, kill, label) — each is a named region of Gray-Scott's parameter space.
PRESETS = (
    (0.0367, 0.0649, "coral"),
    (0.0545, 0.0620, "mitosis"),
    (0.0295, 0.0561, "worms"),
    (0.0250, 0.0600, "solitons"),
    (0.0392, 0.0649, "spirals"),
)

# Simulation steps per rendered frame. The reaction is far slower than 30 fps;
# without this the animation would be imperceptible.
STEPS_PER_FRAME = 14

# Steps run before the first frame, so the clip opens on a formed pattern.
WARM_UP_STEPS = 1400


@register
class ReactionDiffusion(ProceduralGenerator):
    name = "reaction_diffusion"
    description = "Gray-Scott chemistry unfolding slowly"
    loops_natively = False

    def prepare(self, ctx: RenderContext) -> None:
        rng = np.random.default_rng(ctx.seed)
        feed, kill, label = PRESETS[ctx.rng.randrange(len(PRESETS))]

        # Simulated at half the render size: the patterns are smooth, the
        # cost is quadratic, and the upscale costs nothing visually.
        self._h = max(64, ctx.height // 2)
        self._w = max(64, ctx.width // 2)

        self._a = np.ones((self._h, self._w), dtype=np.float32)
        self._b = np.zeros((self._h, self._w), dtype=np.float32)

        # Seed with a handful of blobs; a single one takes too long to spread.
        for _ in range(ctx.rng.randint(4, 9)):
            cy = rng.integers(6, self._h - 6)
            cx = rng.integers(6, self._w - 6)
            r = int(rng.integers(3, 8))
            self._b[cy - r : cy + r, cx - r : cx + r] = 1.0
        self._b += (rng.random((self._h, self._w), dtype=np.float32) * 0.02).astype(np.float32)

        ctx.params.update(
            {
                "preset": label,
                "feed": feed,
                "kill": kill,
                "da": 1.0,
                "db": 0.5,
                "steps_per_frame": STEPS_PER_FRAME,
            }
        )
        self._warm_up(ctx)

    def _warm_up(self, ctx: RenderContext) -> None:
        """Run ahead so frame 0 already has structure instead of grey soup."""
        for _ in range(WARM_UP_STEPS):
            self._step(ctx)

    @staticmethod
    def _laplacian(field: np.ndarray) -> np.ndarray:
        """Nine-point stencil on a torus, weights 0.2 / 0.05 / centre -1.

        These specific weights are not decoration. The naive five-point
        stencil has centre -4, and explicit Euler is only stable while
        ``D·dt/dx² <= 0.25``; with the diffusion rates Gray-Scott is defined
        for (da=1.0) and dt=1 that is violated fourfold. The simulation then
        diverges into a saturated checkerboard within a few hundred steps,
        which the clamp to [0, 1] freezes in place — a still image that looks
        like a working pattern. Normalising the centre to -1 puts the same
        model back inside the stability limit.
        """
        return (
            np.float32(0.2)
            * (
                np.roll(field, 1, axis=0)
                + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1)
                + np.roll(field, -1, axis=1)
            )
            + np.float32(0.05)
            * (
                np.roll(np.roll(field, 1, axis=0), 1, axis=1)
                + np.roll(np.roll(field, 1, axis=0), -1, axis=1)
                + np.roll(np.roll(field, -1, axis=0), 1, axis=1)
                + np.roll(np.roll(field, -1, axis=0), -1, axis=1)
            )
            - field
        )

    def _step(self, ctx: RenderContext) -> None:
        a, b = self._a, self._b
        feed = np.float32(ctx.params["feed"])
        kill = np.float32(ctx.params["kill"])
        reaction = a * b * b
        a += np.float32(ctx.params["da"]) * self._laplacian(a) - reaction + feed * (1.0 - a)
        b += np.float32(ctx.params["db"]) * self._laplacian(b) + reaction - (kill + feed) * b
        np.clip(a, 0.0, 1.0, out=a)
        np.clip(b, 0.0, 1.0, out=b)

    def render(self, frame: int, ctx: RenderContext) -> np.ndarray:
        for _ in range(int(ctx.params["steps_per_frame"])):
            self._step(ctx)

        field = normalise(self._b)
        # Nearest-neighbour upscale back to the render size. The pattern is
        # smooth enough that the ffmpeg lanczos pass afterwards hides it.
        ys = (np.arange(ctx.height) * self._h // ctx.height).clip(0, self._h - 1)
        xs = (np.arange(ctx.width) * self._w // ctx.width).clip(0, self._w - 1)
        return ctx.palette.map(field[np.ix_(ys, xs)], ctx.lut)
