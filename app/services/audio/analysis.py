"""Tempo and beat detection, offline, at upload time.

Trance is the easiest possible case for this: a steady 4/4 with a kick on
every beat, almost always between 128 and 145 BPM. That is why the default
implementation here is about a hundred lines of numpy rather than a
dependency — spectral flux, autocorrelation over a restricted tempo range,
then a phase search for where the beats actually fall.

``librosa`` is used instead when it is installed, because it is more general
and better tested. It is not a requirement: it pulls in scipy, scikit-learn
and numba, which is a lot of machine to add to an image for one number about
music we already know the shape of.

Nothing here runs on the live path. Analysis happens when a file is scanned,
the result is stored, and the broadcast only ever reads it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.core.proc import CommandError, RunOptions, run_bytes

log = get_logger(__name__)

# Analysis sample rate. The kick lives well below 1 kHz, so this is plenty and
# it makes the decode four times cheaper than 44.1 kHz.
SAMPLE_RATE = 22050

# STFT geometry. A 256-sample hop gives ~86 onset measurements per second,
# which resolves a 145 BPM beat (2.4 beats/s) with room to spare.
FRAME = 1024
HOP = 256

# Only this much of the track is analysed. Tempo does not change in trance,
# and reading three minutes of every file makes a library scan glacial.
MAX_ANALYSIS_S = 120.0

# Everything outside this is not trance, and accepting it would mostly mean
# accepting a half- or double-tempo mistake.
DEFAULT_BPM_RANGE = (118.0, 152.0)

# Below this, the estimate is not trustworthy enough to align anything to.
MIN_CONFIDENCE = 0.12

BEATS_PER_BAR = 4

# Onset envelope shorter than this has nothing to detrend or correlate.
_MIN_ENVELOPE_FOR_BASELINE = 32
_MIN_ENVELOPE_FOR_TEMPO = 64
_MIN_ENVELOPE_FOR_PHASE = 8

# Frames per beat below which the pulse train is too coarse to place.
_MIN_PERIOD_FRAMES = 2

# Guard against dividing by a flat parabola when refining the peak.
_PARABOLA_EPSILON = 1e-12

# Width of the moving average that removes slow loudness drift, in frames.
_BASELINE_FRAMES = 31


@dataclass(frozen=True)
class TrackAnalysis:
    bpm: float
    beat_offset_s: float
    confidence: float
    method: str

    @property
    def usable(self) -> bool:
        return self.bpm > 0 and self.confidence >= MIN_CONFIDENCE

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.0

    @property
    def bar_s(self) -> float:
        return self.beat_s * BEATS_PER_BAR

    def as_dict(self) -> dict[str, object]:
        return {
            "bpm": round(self.bpm, 2),
            "beat_offset_s": round(self.beat_offset_s, 4),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "usable": self.usable,
        }


EMPTY = TrackAnalysis(bpm=0.0, beat_offset_s=0.0, confidence=0.0, method="none")


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


def decode_mono(
    path: Path, *, ffmpeg: str = "ffmpeg", seconds: float = MAX_ANALYSIS_S
) -> np.ndarray:
    """Decode the start of a file to mono float32 at SAMPLE_RATE."""
    try:
        raw = run_bytes(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-t",
                f"{seconds:.2f}",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "f32le",
                "-",
            ],
            RunOptions(timeout_s=180.0, check=True),
        )
    except CommandError as exc:
        log.warning("could not decode %s for analysis: %s", path.name, exc)
        return np.zeros(0, dtype=np.float32)

    # Trim to a whole number of float32 samples; a truncated read would
    # otherwise reinterpret the tail as garbage.
    samples = np.frombuffer(raw[: len(raw) // 4 * 4], dtype=np.float32)
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# onset envelope
# --------------------------------------------------------------------------


def onset_envelope(samples: np.ndarray) -> np.ndarray:
    """Spectral flux, weighted towards the bass where the kick lives."""
    if samples.size < FRAME * 4:
        return np.zeros(0, dtype=np.float32)

    frames = 1 + (samples.size - FRAME) // HOP
    window = np.hanning(FRAME).astype(np.float32)
    # A strided view costs no copy; the whole track is one array already.
    strided = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP][:frames]
    spectrum = np.abs(np.fft.rfft(strided * window, axis=1)).astype(np.float32)

    # Emphasise the low end: in trance the beat *is* the kick drum, and the
    # hats and pads above 2 kHz only add noise to the estimate.
    bins = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)
    weight = np.exp(-bins / 400.0).astype(np.float32)
    spectrum *= weight

    # Only increases count: a note starting is an onset, a note ending is not.
    flux = np.diff(spectrum, axis=0)
    envelope = np.maximum(flux, 0.0).sum(axis=1)

    # Remove the slow drift so autocorrelation sees rhythm, not loudness.
    if envelope.size > _MIN_ENVELOPE_FOR_BASELINE:
        kernel = np.ones(_BASELINE_FRAMES, dtype=np.float32) / _BASELINE_FRAMES
        baseline = np.convolve(envelope, kernel, mode="same")
        envelope = np.maximum(envelope - baseline, 0.0)

    peak = float(envelope.max()) if envelope.size else 0.0
    return (envelope / peak).astype(np.float32) if peak > 0 else envelope


# --------------------------------------------------------------------------
# tempo and phase
# --------------------------------------------------------------------------


def _autocorrelation(envelope: np.ndarray) -> np.ndarray:
    padded = np.concatenate([envelope, np.zeros_like(envelope)])
    spectrum = np.fft.rfft(padded)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[: envelope.size]
    return correlation.astype(np.float32)


def estimate_tempo(
    envelope: np.ndarray, bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE
) -> tuple[float, float]:
    """Return (bpm, confidence in 0..1) from the onset envelope."""
    if envelope.size < _MIN_ENVELOPE_FOR_TEMPO:
        return 0.0, 0.0

    frames_per_second = SAMPLE_RATE / HOP
    low_bpm, high_bpm = bpm_range
    min_lag = max(1, round(frames_per_second * 60.0 / high_bpm))
    max_lag = min(envelope.size - 1, round(frames_per_second * 60.0 / low_bpm))
    if max_lag <= min_lag:
        return 0.0, 0.0

    correlation = _autocorrelation(envelope)
    if correlation[0] <= 0:
        return 0.0, 0.0

    window = correlation[min_lag : max_lag + 1]
    # Sum the lag with its first harmonic: a track whose kick lands on every
    # beat correlates at the beat *and* at the bar, and adding them makes the
    # true tempo win over the half-tempo it would otherwise tie with.
    combined = window.copy()
    for lag in range(min_lag, max_lag + 1):
        half = lag // 2
        if min_lag <= half <= max_lag:
            combined[lag - min_lag] += 0.5 * correlation[half]

    best = int(np.argmax(combined)) + min_lag

    # Sub-frame refinement. The autocorrelation is sampled once per 256-sample
    # hop, so at 140 BPM one frame of quantisation is nearly 4 BPM — enough to
    # drift half a beat across a fourteen-second crossfade. Fitting a parabola
    # through the peak and its neighbours recovers the fraction between them.
    #
    # The guard is about the array, not the BPM window. The peak's neighbours
    # exist in the correlation whether or not they fall inside the window, and
    # guarding on the window instead meant that a tempo landing exactly on
    # min_lag skipped refinement and returned the coarse lag. That is not a
    # corner case: at the default 152 BPM ceiling min_lag is 34, which is also
    # the peak for anything near 150, so an entire psytrance library measured
    # as exactly 152.00 — the boundary, reported as if it were a measurement.
    lag = float(best)
    if 0 < best < correlation.size - 1:
        left, centre, right = (
            float(correlation[best - 1]),
            float(correlation[best]),
            float(correlation[best + 1]),
        )
        denominator = left - 2.0 * centre + right
        if abs(denominator) > _PARABOLA_EPSILON:
            shift = 0.5 * (left - right) / denominator
            if -1.0 < shift < 1.0:
                lag = best + shift

    bpm = 60.0 * frames_per_second / lag
    confidence = float(correlation[best] / correlation[0])
    return bpm, max(0.0, min(1.0, confidence))


# Measured systematic bias of the phase estimate against synthetic kick tracks
# at eight tempos: the peak of the onset envelope lands about 32 ms *before*
# the sample where the kick starts, because the flux rises as the transient
# enters the analysis window rather than when it peaks. Correcting the bias is
# worth it; the remaining spread is not reducible by interpolation, which was
# measured and made no difference.
_PHASE_BIAS_S = 0.032

# What the estimate is actually worth, measured the same way: one standard
# deviation is about 27 ms, worst case about 75 ms. At 140 BPM a beat is
# 429 ms, so that is roughly ±6% of a beat typically. Good enough to choose a
# crossfade length in whole bars; *not* good enough to justify trimming audio
# off the front of every track by default.
PHASE_ACCURACY_SD_S = 0.027


def estimate_phase(envelope: np.ndarray, bpm: float) -> float:
    """Seconds from the start of the file to the first beat, modulo one beat.

    Scores every whole-frame offset by how much onset energy its implied pulse
    train collects, and takes the best. Sub-frame interpolation was tried and
    measured: it did not narrow the spread, because the limit is the shape of
    the onset envelope, not the 11.6 ms frame grid.
    """
    if bpm <= 0 or envelope.size < _MIN_ENVELOPE_FOR_PHASE:
        return 0.0
    frames_per_second = SAMPLE_RATE / HOP
    period = frames_per_second * 60.0 / bpm
    if period < _MIN_PERIOD_FRAMES:
        return 0.0

    positions = np.arange(0, envelope.size, period)
    offsets = np.arange(0, round(period))
    scores = np.zeros(offsets.size, dtype=np.float32)
    for i, offset in enumerate(offsets):
        index = np.round(positions + offset).astype(np.int64)
        index = index[index < envelope.size]
        scores[i] = float(envelope[index].sum()) if index.size else 0.0
    if not scores.size or scores.max() <= 0:
        # Nothing landed on any pulse: there is no beat here, so do not report
        # the bias correction as though there were one.
        return 0.0

    seconds = float(np.argmax(scores)) / frames_per_second + _PHASE_BIAS_S
    beat = 60.0 / bpm
    return float(seconds % beat)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def librosa_available() -> bool:
    return importlib.util.find_spec("librosa") is not None


def _analyse_with_librosa(path: Path) -> TrackAnalysis | None:
    try:
        import librosa
    except ImportError:  # pragma: no cover - guarded by librosa_available()
        return None
    try:
        samples, rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=MAX_ANALYSIS_S)
        tempo, beats = librosa.beat.beat_track(y=samples, sr=rate, units="time")
        bpm = float(np.atleast_1d(tempo)[0])
        offset = float(beats[0]) if len(beats) else 0.0
    except Exception as exc:
        log.warning("librosa could not analyse %s: %s", path.name, exc)
        return None
    if bpm <= 0:
        return None
    # librosa gives no confidence figure; a successful beat track on a steady
    # 4/4 is reliable enough to treat as high.
    return TrackAnalysis(bpm=bpm, beat_offset_s=offset, confidence=0.9, method="librosa")


def analyse(
    path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE,
    prefer_librosa: bool = True,
) -> TrackAnalysis:
    """Measure a file's tempo and where its beats fall. Never raises."""
    if not path.is_file():
        return EMPTY

    if prefer_librosa and librosa_available():
        result = _analyse_with_librosa(path)
        if result is not None:
            return result

    samples = decode_mono(path, ffmpeg=ffmpeg)
    if samples.size == 0:
        return EMPTY

    envelope = onset_envelope(samples)
    bpm, confidence = estimate_tempo(envelope, bpm_range)
    if bpm <= 0:
        return EMPTY
    offset = estimate_phase(envelope, bpm)
    return TrackAnalysis(bpm=bpm, beat_offset_s=offset, confidence=confidence, method="numpy")


def bars_near(seconds: float, analysis: TrackAnalysis, *, minimum: float, maximum: float) -> float:
    """The whole number of bars closest to ``seconds``, clamped to the range.

    This is what makes a crossfade musical: eight bars of the outgoing track
    against eight bars of the incoming one, rather than fourteen seconds that
    land wherever they land.
    """
    if not analysis.usable or analysis.bar_s <= 0:
        return seconds
    bars = max(1, round(seconds / analysis.bar_s))
    candidate = bars * analysis.bar_s
    while candidate > maximum and bars > 1:
        bars -= 1
        candidate = bars * analysis.bar_s
    while candidate < minimum:
        bars += 1
        candidate = bars * analysis.bar_s
        if candidate > maximum:
            return max(minimum, min(seconds, maximum))
    return candidate


__all__ = [
    "BEATS_PER_BAR",
    "DEFAULT_BPM_RANGE",
    "EMPTY",
    "MIN_CONFIDENCE",
    "PHASE_ACCURACY_SD_S",
    "SAMPLE_RATE",
    "TrackAnalysis",
    "analyse",
    "bars_near",
    "decode_mono",
    "estimate_phase",
    "estimate_tempo",
    "librosa_available",
    "onset_envelope",
]
