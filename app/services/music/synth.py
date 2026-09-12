"""Sound out of arithmetic: oscillators, envelopes, filters, space.

Everything here is numpy on float64 arrays and nothing else. No scipy, no
model, no GPU — which is the whole point, because the container cannot see
the graphics card and a station that must run on a two-watt NAS cannot
depend on one anyway.

Two rules shape the code:

* **No per-sample Python loops.** A five-minute stereo track is thirteen
  million samples per channel; a loop over those in Python would take longer
  than the track. Anything that has to vary over time varies per *block*.
* **Filters work in the frequency domain.** A resonant filter is recursive by
  nature, and recursion is the one thing numpy cannot vectorise. Shaping the
  spectrum of overlapping windows gives the same sweep for a fraction of the
  effort, and a sweep is what trance is made of.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100

#: Overlapping half-blocks with a Hann window sum back to exactly one, which
#: is what lets the spectral filter reassemble a signal without seams.
FILTER_BLOCK = 1024

#: How many times a moving filter is allowed to step during one buffer. A
#: fixed window size meant a half-second note was filtered in seventy-three
#: steps and a whole track in twenty-six thousand — far past what any ear
#: resolves, and most of the cost of composing. Two dozen updates across a
#: note is already smoother than a hand on a knob.
FILTER_STEPS = 24
MIN_FILTER_BLOCK = 512
MAX_FILTER_BLOCK = 16384

#: Above this many samples a one-shot transform costs more memory than the
#: windowed path saves in time, so the static-filter shortcut steps aside.
#: Forty-five seconds covers every note, chord and drum hit; only whole-track
#: passes are longer.
STATIC_FILTER_LIMIT = 2_000_000

#: Below this a buffer is silence, and scaling it would only amplify noise
#: or divide by zero.
SILENCE = 1e-9


# --------------------------------------------------------------------------
# oscillators
# --------------------------------------------------------------------------


def _phase(freq: np.ndarray | float, samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Running phase in turns, so a waveform is a function of its fraction."""
    if np.isscalar(freq):
        return np.arange(samples, dtype=np.float64) * (float(freq) / sr)
    return np.cumsum(np.asarray(freq, dtype=np.float64) / sr)


def sine(freq: np.ndarray | float, samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.sin(2.0 * np.pi * _phase(freq, samples, sr))


def saw(freq: np.ndarray | float, samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A falling-edge sawtooth in [-1, 1].

    Naive rather than band-limited: the aliasing it folds back sits above the
    lowpass that every voice here runs through, and the character it leaves
    behind is the one the genre is built on.
    """
    turns = _phase(freq, samples, sr)
    return 2.0 * (turns - np.floor(turns)) - 1.0


def square(freq: np.ndarray | float, samples: int, duty: float = 0.5, sr: int = SAMPLE_RATE):
    return np.where((_phase(freq, samples, sr) % 1.0) < duty, 1.0, -1.0)


def supersaw(
    freq: float,
    samples: int,
    *,
    voices: int = 7,
    detune_cents: float = 18.0,
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """The sound of the genre: one note played by several slightly wrong copies.

    The beating between voices a few cents apart is the whole effect, so they
    are spread evenly rather than randomly and each starts at its own phase —
    identical phases would give one loud saw and a click instead of a chorus.
    """
    return (
        _saw_stack(freq, samples, voices=voices, detune_cents=detune_cents, rng=rng, sr=sr).sum(
            axis=0
        )
        / voices
    )


def _saw_stack(
    freq: float,
    samples: int,
    *,
    voices: int,
    detune_cents: float,
    rng: np.random.Generator,
    sr: int,
) -> np.ndarray:
    """Every detuned copy at once, as a (voices, samples) block.

    A Python loop over seven voices was costing more than the arithmetic it
    contained — and the melody alone asks for this a few hundred times per
    track. One broadcast does the lot.
    """
    offsets = np.linspace(-1.0, 1.0, voices)
    ratios = 2.0 ** (offsets * detune_cents / 1200.0)
    steps = (freq * ratios / sr)[:, np.newaxis]
    # Each copy starts somewhere else: identical phases would give one loud
    # saw and a click instead of a chorus.
    starts = rng.random(voices)[:, np.newaxis]
    phase = starts + np.arange(samples, dtype=np.float64)[np.newaxis, :] * steps
    # floor rather than %: the modulo goes through fmod and costs about half
    # again as much for an answer that is identical on positive input, and
    # this runs over a million samples a few hundred times per track.
    return 2.0 * (phase - np.floor(phase)) - 1.0


def supersaw_stereo(
    freq: float,
    samples: int,
    *,
    voices: int = 7,
    detune_cents: float = 18.0,
    spread: float = 0.85,
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """The same stack, with the detuned copies placed across the stereo field.

    Width by delaying one channel a few samples is the cheap trick, and it
    collapses the moment anything sums to mono — which a phone, a club system
    and half of YouTube's listeners all do. Panning the individual voices is
    how the sound was made in the first place: each copy is a real, different
    waveform, so the width survives summing and the centre stays solid.
    """
    stack = _saw_stack(freq, samples, voices=voices, detune_cents=detune_cents, rng=rng, sr=sr)
    # Equal power, so the middle voice is no louder than the edges.
    angles = (np.linspace(-1.0, 1.0, voices) * spread + 1.0) * 0.25 * np.pi
    left = (stack * np.cos(angles)[:, np.newaxis]).sum(axis=0) / voices
    right = (stack * np.sin(angles)[:, np.newaxis]).sum(axis=0) / voices
    return left, right


def noise(samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(samples)


# --------------------------------------------------------------------------
# envelopes
# --------------------------------------------------------------------------


def adsr(
    samples: int,
    *,
    attack_s: float = 0.005,
    decay_s: float = 0.1,
    sustain: float = 0.7,
    release_s: float = 0.2,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """A classic four-stage envelope, sized to fit whatever room it is given."""
    attack = max(1, int(attack_s * sr))
    decay = max(1, int(decay_s * sr))
    release = max(1, int(release_s * sr))
    hold = max(0, samples - attack - decay - release)

    pieces = [
        np.linspace(0.0, 1.0, attack, endpoint=False),
        np.linspace(1.0, sustain, decay, endpoint=False),
        np.full(hold, sustain),
        np.linspace(sustain, 0.0, release),
    ]
    env = np.concatenate(pieces)
    return env[:samples] if env.size >= samples else np.pad(env, (0, samples - env.size))


def percussive(samples: int, decay_s: float, *, curve: float = 1.0, sr: int = SAMPLE_RATE):
    """Instant attack, exponential fall — drums, plucks, anything struck."""
    t = np.arange(samples, dtype=np.float64) / sr
    return np.exp(-t / max(1e-4, decay_s)) ** curve


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------


def _response(freqs: np.ndarray, cutoff: float, resonance: float, order: int) -> np.ndarray:
    """The magnitude a lowpass should have, resonant peak included.

    Built directly rather than derived from a difference equation: we are
    multiplying a spectrum, so the shape *is* the filter.
    """
    ratio = np.maximum(freqs, 1e-6) / max(20.0, cutoff)
    magnitude = 1.0 / np.sqrt(1.0 + ratio ** (2 * order))
    if resonance > 0.0:
        # A bump one octave wide sitting on the corner. Without it a sweep
        # sounds like a blanket being pulled off rather than an instrument.
        peak = np.exp(-((np.log2(np.maximum(ratio, 1e-6))) ** 2) / 0.08)
        magnitude = magnitude + resonance * peak
    return magnitude


def sweep_lowpass(
    signal: np.ndarray,
    cutoff: np.ndarray | float,
    *,
    resonance: float = 0.0,
    order: int = 2,
    sr: int = SAMPLE_RATE,
    block: int | None = None,
) -> np.ndarray:
    """Lowpass whose corner moves, applied by reshaping overlapping windows.

    `cutoff` is either one frequency or a curve as long as the signal; only
    its value at the middle of each window is used, so a sweep is stepped at
    about ninety times a second. Far finer than the ear resolves a filter
    movement, and it costs two transforms per window instead of a recursion
    nothing can vectorise.
    """
    n = signal.size
    if n == 0:
        return signal

    if np.isscalar(cutoff) and n <= STATIC_FILTER_LIMIT:
        # A corner that never moves needs no windows: one transform over the
        # whole buffer does what several hundred overlapping ones would, and
        # every note of every arpeggio comes through here. Long buffers stay
        # on the block path, where the transform size is bounded.
        spectrum = np.fft.rfft(signal)
        spectrum *= _response(np.fft.rfftfreq(n, 1.0 / sr), float(cutoff), resonance, order)
        return np.fft.irfft(spectrum, n)

    if block is None:
        # Scale the window to the buffer. A corner that does not move needs no
        # time resolution at all, so those get the widest window allowed.
        wanted = n if np.isscalar(cutoff) else n // FILTER_STEPS
        block = int(2 ** round(np.log2(max(MIN_FILTER_BLOCK, min(MAX_FILTER_BLOCK, wanted)))))

    hop = block // 2
    window = np.hanning(block + 1)[:block]
    padded = np.pad(signal, (hop, block))
    out = np.zeros(padded.size, dtype=np.float64)
    freqs = np.fft.rfftfreq(block, 1.0 / sr)

    curve = np.full(n, float(cutoff)) if np.isscalar(cutoff) else np.asarray(cutoff, float)

    for start in range(0, padded.size - block, hop):
        centre = min(n - 1, max(0, start - hop + hop // 2))
        shaped = np.fft.rfft(padded[start : start + block] * window)
        shaped *= _response(freqs, float(curve[centre]), resonance, order)
        out[start : start + block] += np.fft.irfft(shaped, block)

    return out[hop : hop + n]


def highpass(signal: np.ndarray, cutoff: float, *, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Everything the matching lowpass would have thrown away."""
    return signal - sweep_lowpass(signal, cutoff, order=2, sr=sr)


def bandpass(signal: np.ndarray, low: float, high: float, *, sr: int = SAMPLE_RATE):
    return highpass(sweep_lowpass(signal, high, sr=sr), low, sr=sr)


# --------------------------------------------------------------------------
# space
# --------------------------------------------------------------------------


def reverb(
    signal: np.ndarray,
    *,
    rng: np.random.Generator,
    seconds: float = 1.8,
    taps: int = 28,
    damping: float = 4500.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Room as a handful of decaying echoes rather than a feedback network.

    A comb-and-allpass reverb is recursive, which here means a Python loop over
    every sample. A scattering of taps whose spacing never repeats gives a
    convincing tail with nothing but shifted addition, and the ear cannot pick
    the difference out from under a kick drum.
    """
    out = np.zeros_like(signal)
    delays = np.sort(rng.uniform(0.012, seconds, taps))
    for delay in delays:
        offset = int(delay * sr)
        if offset >= signal.size:
            continue
        gain = 10 ** (-3.0 * delay / seconds)  # -60 dB by the end of the tail
        out[offset:] += signal[: signal.size - offset] * gain * rng.uniform(0.6, 1.0)
    damped = sweep_lowpass(out, damping, sr=sr)
    del out
    damped /= max(1.0, taps * 0.25)
    return damped.astype(signal.dtype, copy=False)


def delay_line(
    signal: np.ndarray,
    *,
    time_s: float,
    feedback: float = 0.42,
    repeats: int = 8,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Echo, unrolled.

    The feedback loop is written out as a fixed number of taps because the
    gain of each repeat is known in advance — there is no reason to compute
    recursively what closes to a geometric series.
    """
    out = np.zeros_like(signal)
    step = max(1, int(time_s * sr))
    for repeat in range(1, repeats + 1):
        offset = repeat * step
        if offset >= signal.size:
            break
        out[offset:] += signal[: signal.size - offset] * (feedback**repeat)
    return out


def sidechain(samples: int, hits: np.ndarray, *, depth: float = 0.75, release_s: float = 0.28):
    """The pump: everything ducks out of the kick's way and swells back.

    Trance without this sounds like a wall. It is the single cheapest thing
    that makes a mix breathe, and it is one exponential per kick.
    """
    env = np.ones(samples, dtype=np.float64)
    length = int(release_s * SAMPLE_RATE)
    shape = 1.0 - depth * np.exp(-np.linspace(0.0, 5.0, length))
    for hit in hits:
        start = int(hit)
        if start >= samples:
            break
        end = min(samples, start + length)
        env[start:end] = np.minimum(env[start:end], shape[: end - start])
    return env


# --------------------------------------------------------------------------
# mixing
# --------------------------------------------------------------------------


def place(buffer: np.ndarray, signal: np.ndarray, at: int) -> None:
    """Add a voice into the mix at a sample offset, clipped to the buffer."""
    if at >= buffer.size:
        return
    start = max(0, at)
    head = start - at
    usable = min(signal.size - head, buffer.size - start)
    if usable > 0:
        buffer[start : start + usable] += signal[head : head + usable]


def place_stereo(
    left_buffer: np.ndarray,
    right_buffer: np.ndarray,
    voice: tuple[np.ndarray, np.ndarray],
    at: int,
) -> None:
    place(left_buffer, voice[0], at)
    place(right_buffer, voice[1], at)


def stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1)


def widen(signal: np.ndarray, amount: float, *, sr: int = SAMPLE_RATE) -> tuple:
    """Two channels from one, by delaying a few samples — the Haas trick."""
    offset = max(1, int(amount * 0.02 * sr))
    left = signal
    right = np.concatenate([np.zeros(offset), signal[:-offset]]) if offset < signal.size else signal
    return left, right


def normalise(signal: np.ndarray, peak: float = 0.89) -> np.ndarray:
    """Scale to a target peak, and never divide by a silence."""
    loudest = float(np.max(np.abs(signal))) if signal.size else 0.0
    return signal * (peak / loudest) if loudest > SILENCE else signal


def soft_clip(signal: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Keep the peaks in bounds without the crackle of hard clipping."""
    return np.tanh(signal * drive) / np.tanh(drive)


__all__ = [
    "FILTER_BLOCK",
    "SAMPLE_RATE",
    "SILENCE",
    "STATIC_FILTER_LIMIT",
    "adsr",
    "bandpass",
    "delay_line",
    "highpass",
    "noise",
    "normalise",
    "percussive",
    "place",
    "place_stereo",
    "reverb",
    "saw",
    "sidechain",
    "sine",
    "soft_clip",
    "square",
    "stereo",
    "supersaw",
    "supersaw_stereo",
    "sweep_lowpass",
    "widen",
]
