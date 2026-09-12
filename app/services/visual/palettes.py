"""Colour palettes: JSON in, 256-entry lookup table out.

Mapping a scalar field to colour is the single hottest operation in the
generator — it runs on every pixel of every frame — so it is done by indexing
a precomputed LUT rather than by interpolating per pixel.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)

LUT_SIZE = 256

# "#abc" and "#aabbcc" are the two accepted spellings.
_SHORT_HEX, _FULL_HEX = 3, 6
# A gradient needs somewhere to go.
MIN_PALETTE_COLORS = 2

# Used only when palettes.json is missing or unreadable, so that a generator
# never fails for want of colours.
FALLBACK_COLORS = ["#02040f", "#062044", "#0b4f8a", "#18a3c9", "#5ee7f0", "#c8fbff"]

# Keyed by (path, mtime) so an edited palettes.json is picked up without a
# restart, but an unchanged one is not re-parsed on every request.
_PALETTE_CACHE: dict[tuple[str, int], PaletteSet] = {}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == _SHORT_HEX:
        text = "".join(c * 2 for c in text)
    if len(text) != _FULL_HEX:
        raise ValueError(f"{value!r} is not a #rrggbb colour")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


@dataclass(frozen=True)
class Palette:
    id: str
    name: str
    colors: tuple[str, ...]
    weight: float = 1.0

    def lut(self) -> np.ndarray:
        """A (256, 3) uint8 ramp interpolating the stops evenly."""
        stops = np.array([hex_to_rgb(c) for c in self.colors], dtype=np.float32)
        if len(stops) == 1:
            return np.repeat(stops.astype(np.uint8), LUT_SIZE, axis=0)
        positions = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
        targets = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
        channels = [np.interp(targets, positions, stops[:, i]) for i in range(3)]
        return np.clip(np.stack(channels, axis=1), 0, 255).astype(np.uint8)

    def map(self, values: np.ndarray, lut: np.ndarray | None = None) -> np.ndarray:
        """Colour a float field in [0, 1]. Out-of-range values are clamped.

        NaNs are mapped to the darkest stop rather than allowed to become a
        garbage index — a single NaN from a diverging simulation would
        otherwise scatter random pixels across the frame.
        """
        table = self.lut() if lut is None else lut
        scaled = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        indices = np.clip(scaled * (LUT_SIZE - 1), 0, LUT_SIZE - 1).astype(np.uint8)
        return table[indices]


class PaletteSet:
    """The palettes available to the generator, loaded from config."""

    def __init__(self, palettes: list[Palette]) -> None:
        if not palettes:
            palettes = [Palette(id="fallback", name="Fallback", colors=tuple(FALLBACK_COLORS))]
        self._palettes = palettes

    def __len__(self) -> int:
        return len(self._palettes)

    def __iter__(self):
        return iter(self._palettes)

    @property
    def all(self) -> list[Palette]:
        return list(self._palettes)

    def by_id(self, palette_id: str) -> Palette | None:
        return next((p for p in self._palettes if p.id == palette_id), None)

    def choose(self, rng: random.Random) -> Palette:
        weights = [max(0.0, p.weight) for p in self._palettes]
        if sum(weights) <= 0:
            return rng.choice(self._palettes)
        return rng.choices(self._palettes, weights=weights, k=1)[0]

    @classmethod
    def load(cls, path: Path) -> PaletteSet:
        """Read palettes.json, reusing the parse until the file changes."""
        try:
            stamp = (str(path), path.stat().st_mtime_ns)
        except OSError:
            stamp = (str(path), 0)
        cached = _PALETTE_CACHE.get(stamp)
        if cached is not None:
            return cached
        result = cls._load_uncached(path)
        _PALETTE_CACHE.clear()  # only the current file is worth keeping
        _PALETTE_CACHE[stamp] = result
        return result

    @classmethod
    def _load_uncached(cls, path: Path) -> PaletteSet:
        if not path.is_file():
            log.warning("no palette file at %s; using the built-in fallback", path)
            return cls([])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("could not read %s (%s); using the built-in fallback", path, exc)
            return cls([])

        palettes: list[Palette] = []
        for entry in data.get("palettes", []):
            if not entry.get("enabled", True):
                continue
            colors = tuple(entry.get("colors") or ())
            if len(colors) < MIN_PALETTE_COLORS:
                log.warning("palette %r has fewer than two colours; skipped", entry.get("id"))
                continue
            try:
                for color in colors:
                    hex_to_rgb(color)
            except ValueError as exc:
                log.warning("palette %r has a bad colour (%s); skipped", entry.get("id"), exc)
                continue
            palettes.append(
                Palette(
                    id=str(entry.get("id") or entry.get("name") or f"palette{len(palettes)}"),
                    name=str(entry.get("name") or entry.get("id") or "Untitled"),
                    colors=colors,
                    weight=float(entry.get("weight", 1.0)),
                )
            )
        log.info("loaded %d palettes from %s", len(palettes), path)
        return cls(palettes)


__all__ = ["LUT_SIZE", "Palette", "PaletteSet", "hex_to_rgb"]
