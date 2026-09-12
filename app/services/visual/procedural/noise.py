"""Tileable value noise and fBm, vectorised over whole frames.

Everything here is a pure function of integer lattice coordinates and a seed,
which buys two properties the rest of the generator depends on:

* **Determinism** — the same seed rebuilds the same clip, so a block's sidecar
  is enough to reproduce it.
* **Tileability** — the lattice wraps on a period, so sampling time over
  exactly one period produces a *perfect loop* with no crossfade needed. That
  is why the third dimension exists: it is time.
"""

from __future__ import annotations

import functools

import numpy as np

# Classic Perlin-style permutation table. Three uint8 gathers replace the
# arithmetic hash this used to do, and the difference is not marginal: the
# noise-heavy generators went from 0.04x real time to usable on the strength
# of this one change. numpy indexing is simply much faster than a chain of
# 64-bit multiplies and shifts over a 230k-element array.
_PERM_SIZE = 256


@functools.lru_cache(maxsize=64)
def _permutation(seed: int) -> np.ndarray:
    """A doubled 0..255 permutation, so ``+1`` needs no second mask."""
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    table = rng.permutation(_PERM_SIZE).astype(np.uint8)
    return np.concatenate([table, table])


def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Map an integer lattice point to a float in [0, 1)."""
    perm = _permutation(seed)
    a = perm[(ix & 255).astype(np.uint8)].astype(np.int16)
    b = perm[(a + (iy & 255).astype(np.int16)) & 255].astype(np.int16)
    c = perm[(b + (iz & 255).astype(np.int16)) & 255]
    return c.astype(np.float32) * np.float32(1.0 / _PERM_SIZE)


def _corner_hashes(
    x: tuple[np.ndarray, np.ndarray],
    y: tuple[np.ndarray, np.ndarray],
    z: tuple[np.ndarray, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, ...]:
    """The eight lattice corners, sharing work between them.

    Done naively this is eight independent three-level lookups: 24 gathers
    over a full-frame array. The x level depends only on x, and the xy level
    only on x and y, so hoisting them out gives 2 + 4 + 8 = 14 instead — for
    identical results.
    """
    perm = _permutation(seed)
    scale = np.float32(1.0 / _PERM_SIZE)
    x0, x1 = x
    y0, y1 = y
    z0, z1 = z

    px0 = perm[(x0 & 255).astype(np.uint8)].astype(np.int16)
    px1 = perm[(x1 & 255).astype(np.uint8)].astype(np.int16)

    iy0 = (y0 & 255).astype(np.int16)
    iy1 = (y1 & 255).astype(np.int16)
    p00 = perm[(px0 + iy0) & 255].astype(np.int16)
    p10 = perm[(px1 + iy0) & 255].astype(np.int16)
    p01 = perm[(px0 + iy1) & 255].astype(np.int16)
    p11 = perm[(px1 + iy1) & 255].astype(np.int16)

    iz0 = (z0 & 255).astype(np.int16)
    iz1 = (z1 & 255).astype(np.int16)
    return tuple(
        perm[(level + offset) & 255].astype(np.float32) * scale
        for offset in (iz0, iz1)
        for level in (p00, p10, p01, p11)
    )


def _smootherstep(t: np.ndarray) -> np.ndarray:
    """Ken Perlin's quintic curve: zero first *and* second derivative at the
    ends, which is what keeps fBm from showing lattice-aligned creases."""
    return t * t * t * (t * (t * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0))


def value_noise_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: float | np.ndarray,
    *,
    period: tuple[int, int, int],
    seed: int = 0,
) -> np.ndarray:
    """Trilinearly interpolated value noise in [0, 1), tiling on ``period``."""
    px, py, pz = (max(1, int(p)) for p in period)

    z_arr = np.asarray(z, dtype=np.float32)
    x0f, y0f, z0f = np.floor(x), np.floor(y), np.floor(z_arr)
    tx = _smootherstep((x - x0f).astype(np.float32))
    ty = _smootherstep((y - y0f).astype(np.float32))
    tz = _smootherstep((z_arr - z0f).astype(np.float32))

    # Wrapping the lattice indices is the whole tileability trick.
    x0 = np.mod(x0f.astype(np.int64), px)
    y0 = np.mod(y0f.astype(np.int64), py)
    z0 = np.mod(z0f.astype(np.int64), pz)
    x1, y1, z1 = (x0 + 1) % px, (y0 + 1) % py, (z0 + 1) % pz

    c000, c100, c010, c110, c001, c101, c011, c111 = _corner_hashes(
        (x0, x1), (y0, y1), (z0, z1), seed
    )

    c00 = c000 + (c100 - c000) * tx
    c10 = c010 + (c110 - c010) * tx
    c01 = c001 + (c101 - c001) * tx
    c11 = c011 + (c111 - c011) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    return c0 + (c1 - c0) * tz


def fbm3(
    x: np.ndarray,
    y: np.ndarray,
    z: float | np.ndarray,
    *,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    period: tuple[int, int, int] = (16, 16, 16),
    seed: int = 0,
) -> np.ndarray:
    """Fractional Brownian motion, normalised to roughly [0, 1].

    The period is multiplied by the lacunarity at every octave so that each
    octave tiles on the same overall span; skipping that is the usual reason
    a "looping" animation visibly jumps.
    """
    total = np.zeros_like(x, dtype=np.float32)
    amplitude = np.float32(1.0)
    total_amplitude = np.float32(0.0)
    frequency = np.float32(1.0)
    px, py, pz = period

    for octave in range(max(1, octaves)):
        total += amplitude * value_noise_3d(
            x * frequency,
            y * frequency,
            np.asarray(z, dtype=np.float32) * frequency,
            period=(
                max(1, round(px * frequency)),
                max(1, round(py * frequency)),
                max(1, round(pz * frequency)),
            ),
            seed=seed + octave * 1013,
        )
        total_amplitude += amplitude
        amplitude *= np.float32(gain)
        frequency *= np.float32(lacunarity)

    return total / max(total_amplitude, np.float32(1e-6))


def loop_angle(frame: int, total_frames: int) -> float:
    """Position within a clip as an angle, so any sin/cos of it loops."""
    return 2.0 * np.pi * (frame / max(1, total_frames))


def pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised coordinates with a square aspect: y in [-1, 1], x scaled.

    Returned as ``(x, y)`` float32 arrays shaped (height, width).
    """
    aspect = width / max(1, height)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xs = np.linspace(-aspect, aspect, width, dtype=np.float32)
    return np.meshgrid(xs, ys)


__all__ = ["fbm3", "loop_angle", "pixel_grid", "value_noise_3d"]
