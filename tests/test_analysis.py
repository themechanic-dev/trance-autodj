"""Tempo and beat detection, and the beat-aligned crossfade it feeds."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.audio import analysis
from app.services.audio.analysis import (
    BEATS_PER_BAR,
    TrackAnalysis,
    analyse,
    bars_near,
    estimate_phase,
    estimate_tempo,
    onset_envelope,
)

SR = 44100


def kick_track(path: Path, bpm: float, *, seconds: float = 30.0, offset_s: float = 0.0) -> Path:
    """A trance-shaped test signal: a decaying 55 Hz thump on every beat."""
    total = int(seconds * SR)
    samples = np.zeros(total, dtype=np.float32)
    beat = 60.0 / bpm

    envelope_t = np.arange(round(0.12 * SR)) / SR
    kick = (np.sin(2 * np.pi * 55 * envelope_t) * np.exp(-envelope_t * 28)).astype(np.float32)

    position = offset_s
    while position * SR + kick.size < total:
        start = round(position * SR)
        samples[start : start + kick.size] += kick
        position += beat

    # A quiet pad, so it is not a bare impulse train.
    samples += 0.05 * np.sin(2 * np.pi * 220 * np.arange(total) / SR).astype(np.float32)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(samples, -1, 1) * 32000).astype("<i2").tobytes())
    return path


@pytest.mark.slow
@pytest.mark.parametrize("bpm", [128, 132, 138, 140, 145])
def test_tempo_is_measured_accurately(tmp_path: Path, bpm: int):
    """Measured against synthetic kicks: the error stays well under 1 BPM.

    That matters because the error accumulates across a crossfade — at 1 BPM
    out, two tracks drift a quarter of a beat apart over fourteen seconds.
    """
    import shutil

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    result = analyse(kick_track(tmp_path / f"{bpm}.wav", bpm), prefer_librosa=False)
    assert result.usable
    assert abs(result.bpm - bpm) < 1.0, f"measured {result.bpm:.2f} for {bpm}"


@pytest.mark.slow
def test_beat_phase_is_found(tmp_path: Path):
    import shutil

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    bpm, offset = 140.0, 0.2
    result = analyse(kick_track(tmp_path / "phase.wav", bpm, offset_s=offset), prefer_librosa=False)
    beat = 60.0 / bpm
    error = (result.beat_offset_s - offset) % beat
    error = min(error, beat - error)
    # Measured spread is about 27 ms; a beat here is 429 ms.
    assert error < 0.08, f"phase off by {error * 1000:.0f} ms"


@pytest.mark.slow
def test_a_file_that_cannot_be_decoded_is_not_an_error(tmp_path: Path):
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"this is not audio")
    assert analyse(broken, prefer_librosa=False) is not None
    assert not analyse(broken, prefer_librosa=False).usable


def test_a_missing_file_returns_the_empty_result(tmp_path: Path):
    assert analyse(tmp_path / "nope.mp3") == analysis.EMPTY


def test_silence_produces_no_tempo():
    assert estimate_tempo(np.zeros(200, dtype=np.float32)) == (0.0, 0.0)
    assert estimate_phase(np.zeros(200, dtype=np.float32), 140.0) == 0.0


def test_onset_envelope_is_normalised_and_finite():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(analysis.SAMPLE_RATE * 3).astype(np.float32) * 0.1
    envelope = onset_envelope(samples)
    assert envelope.size > 0
    assert np.isfinite(envelope).all()
    assert envelope.min() >= 0.0 and envelope.max() <= 1.0


def test_a_very_short_signal_is_handled():
    assert onset_envelope(np.zeros(100, dtype=np.float32)).size == 0


def test_the_tempo_range_is_respected():
    """Accepting anything outside trance tempo mostly means accepting a
    half- or double-tempo mistake."""
    rng = np.random.default_rng(1)
    envelope = rng.random(4000).astype(np.float32)
    bpm, _ = estimate_tempo(envelope, (118.0, 152.0))
    assert bpm == 0.0 or 117.0 <= bpm <= 153.0


# --- derived values ------------------------------------------------------


def analysis_at(bpm: float) -> TrackAnalysis:
    return TrackAnalysis(bpm=bpm, beat_offset_s=0.0, confidence=0.9, method="test")


def test_bar_and_beat_lengths():
    result = analysis_at(120.0)
    assert result.beat_s == pytest.approx(0.5)
    assert result.bar_s == pytest.approx(2.0)
    assert BEATS_PER_BAR == 4


@pytest.mark.parametrize("bpm", [128.0, 132.0, 138.0, 140.0, 145.0])
def test_crossfade_lands_on_a_whole_number_of_bars(bpm: float):
    result = analysis_at(bpm)
    seconds = bars_near(14.0, result, minimum=4.0, maximum=30.0)
    bars = seconds / result.bar_s
    assert bars == pytest.approx(round(bars)), f"{seconds}s is {bars} bars at {bpm}"
    assert abs(seconds - 14.0) <= result.bar_s


def test_bars_near_respects_the_configured_bounds():
    result = analysis_at(140.0)
    assert bars_near(14.0, result, minimum=4.0, maximum=8.0) <= 8.0
    assert bars_near(1.0, result, minimum=4.0, maximum=30.0) >= 4.0


def test_an_unusable_measurement_leaves_the_duration_alone():
    poor = TrackAnalysis(bpm=140.0, beat_offset_s=0.0, confidence=0.01, method="test")
    assert bars_near(14.0, poor, minimum=4.0, maximum=30.0) == 14.0
    assert bars_near(14.0, analysis.EMPTY, minimum=4.0, maximum=30.0) == 14.0


def test_usable_needs_both_a_tempo_and_confidence():
    assert analysis_at(140.0).usable
    assert not TrackAnalysis(bpm=0.0, beat_offset_s=0.0, confidence=0.9, method="x").usable
    assert not TrackAnalysis(bpm=140.0, beat_offset_s=0.0, confidence=0.0, method="x").usable


@pytest.mark.slow
@pytest.mark.parametrize("bpm", [148, 150, 151])
def test_a_tempo_at_the_top_of_the_range_is_still_measured(tmp_path: Path, bpm: int):
    """The sub-frame refinement used to be skipped when the peak landed on the
    edge of the search window, and at the default 152 BPM ceiling that edge is
    lag 34 — the peak for anything near 150. Every track in a psytrance
    library came back as exactly 152.00: the boundary, reported as a
    measurement, with no hint that it was one.
    """
    import shutil

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    result = analyse(kick_track(tmp_path / f"{bpm}.wav", bpm), prefer_librosa=False)
    assert result.usable
    assert result.bpm != 152.0, "that is the ceiling, not a measurement"
    assert abs(result.bpm - bpm) < 1.0, f"measured {result.bpm:.2f} for {bpm}"
