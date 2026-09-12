"""Palette loading and colour mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.visual.palettes import LUT_SIZE, Palette, PaletteSet, hex_to_rgb


def test_hex_parsing():
    assert hex_to_rgb("#ff8800") == (255, 136, 0)
    assert hex_to_rgb("ff8800") == (255, 136, 0)
    assert hex_to_rgb("#f80") == (255, 136, 0)


def test_bad_hex_is_rejected():
    with pytest.raises(ValueError):
        hex_to_rgb("#12345")


def test_lut_spans_the_stops():
    palette = Palette(id="t", name="T", colors=("#000000", "#ffffff"))
    lut = palette.lut()
    assert lut.shape == (LUT_SIZE, 3)
    assert tuple(lut[0]) == (0, 0, 0)
    assert tuple(lut[-1]) == (255, 255, 255)
    assert lut.dtype == np.uint8


def test_map_clamps_out_of_range_values():
    palette = Palette(id="t", name="T", colors=("#000000", "#ffffff"))
    result = palette.map(np.array([[-5.0, 0.5, 5.0]], dtype=np.float32))
    assert tuple(result[0, 0]) == (0, 0, 0)
    assert tuple(result[0, 2]) == (255, 255, 255)


def test_map_survives_nan_and_inf():
    """A diverging simulation must not scatter random pixels into the frame."""
    palette = Palette(id="t", name="T", colors=("#000000", "#ffffff"))
    values = np.array([[np.nan, np.inf, -np.inf]], dtype=np.float32)
    result = palette.map(values)
    assert result.dtype == np.uint8
    assert tuple(result[0, 0]) == (0, 0, 0)
    assert tuple(result[0, 1]) == (255, 255, 255)


def test_single_colour_palette_is_flat_but_valid():
    palette = Palette(id="t", name="T", colors=("#123456",))
    assert palette.lut().shape == (LUT_SIZE, 3)


def test_load_reads_the_shipped_file():
    palettes = PaletteSet.load(Path(__file__).resolve().parent.parent / "config" / "palettes.json")
    assert len(palettes) >= 5
    assert palettes.by_id("deep_ocean") is not None


def test_missing_file_falls_back_instead_of_failing(tmp_path: Path):
    palettes = PaletteSet.load(tmp_path / "nope.json")
    assert len(palettes) == 1
    assert palettes.all[0].lut().shape == (LUT_SIZE, 3)


def test_broken_json_falls_back(tmp_path: Path):
    path = tmp_path / "palettes.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(PaletteSet.load(path)) == 1


def test_disabled_and_invalid_entries_are_skipped(tmp_path: Path):
    path = tmp_path / "palettes.json"
    path.write_text(
        json.dumps(
            {
                "palettes": [
                    {"id": "ok", "colors": ["#000000", "#ffffff"]},
                    {"id": "off", "colors": ["#000000", "#ffffff"], "enabled": False},
                    {"id": "onecolour", "colors": ["#000000"]},
                    {"id": "badhex", "colors": ["#000000", "nonsense"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    palettes = PaletteSet.load(path)
    assert [p.id for p in palettes] == ["ok"]


def test_choose_respects_weights(tmp_path: Path):
    import random

    path = tmp_path / "palettes.json"
    path.write_text(
        json.dumps(
            {
                "palettes": [
                    {"id": "likely", "colors": ["#000000", "#ffffff"], "weight": 20},
                    {"id": "rare", "colors": ["#000000", "#111111"], "weight": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    palettes = PaletteSet.load(path)
    rng = random.Random(5)
    picks = [palettes.choose(rng).id for _ in range(400)]
    assert picks.count("likely") > picks.count("rare") * 5


def test_the_cache_notices_an_edited_file(tmp_path: Path):
    import os
    import time

    path = tmp_path / "palettes.json"
    path.write_text(json.dumps({"palettes": [{"id": "a", "colors": ["#000", "#fff"]}]}))
    assert [p.id for p in PaletteSet.load(path)] == ["a"]

    time.sleep(0.01)
    path.write_text(json.dumps({"palettes": [{"id": "b", "colors": ["#000", "#fff"]}]}))
    os.utime(path, None)
    assert [p.id for p in PaletteSet.load(path)] == ["b"]
