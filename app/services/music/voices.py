"""The instruments: one function per sound, each returning a mono array.

A voice knows its own shape and nothing about the arrangement — no bar, no
tempo, no position. The composer decides when a voice speaks and how loud;
everything here answers only "what does it sound like".

Tuning is equal temperament from A4 = 440 Hz, which is why every voice takes
a frequency rather than a note name: transposing a whole track is then one
multiplication instead of a lookup table.
"""

from __future__ import annotations

import numpy as np

from app.services.music import synth
from app.services.music.synth import SAMPLE_RATE

#: Halfway through a roll the steps halve, which is what makes it read as an
#: approach rather than a pattern.
ROLL_TIGHTENS_AT = 0.5


def hz(midi: float) -> float:
    """Concert pitch from a MIDI note number. A4 = 69 = 440 Hz."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# --------------------------------------------------------------------------
# drums
# --------------------------------------------------------------------------


def kick(*, decay_s: float = 0.42, punch: float = 1.0) -> np.ndarray:
    """Four on the floor, and the thing everything else ducks out of the way of.

    A kick is a sine whose pitch collapses: it starts near 150 Hz so the ear
    hears an attack, and lands on 48 Hz within a twentieth of a second so what
    is left is weight. Holding the pitch still gives a beep; dropping it too
    slowly gives a tom.
    """
    samples = int(decay_s * SAMPLE_RATE)
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    pitch = 48.0 + 102.0 * np.exp(-t / 0.022)
    body = np.sin(2.0 * np.pi * np.cumsum(pitch) / SAMPLE_RATE)
    body *= synth.percussive(samples, decay_s * 0.42)

    # The click is what survives a phone speaker, where the 48 Hz does not.
    click_len = int(0.006 * SAMPLE_RATE)
    click = np.zeros(samples)
    click[:click_len] = np.linspace(1.0, 0.0, click_len) ** 2
    return synth.soft_clip(body * 1.15 + click * 0.35 * punch, 1.4)


def clap(rng: np.random.Generator, *, spread_s: float = 0.011) -> np.ndarray:
    """Beats two and four. Several bursts in quick succession, then a room.

    One burst reads as a snare; three staggered by about ten milliseconds read
    as hands, which is the sound the genre uses.
    """
    samples = int(0.34 * SAMPLE_RATE)
    out = np.zeros(samples)
    for index in range(3):
        at = int(index * spread_s * SAMPLE_RATE)
        burst = synth.noise(int(0.05 * SAMPLE_RATE), rng)
        burst *= synth.percussive(burst.size, 0.011)
        synth.place(out, burst, at)
    tail = synth.noise(samples, rng) * synth.percussive(samples, 0.09)
    return synth.bandpass(out * 0.9 + tail * 0.35, 900.0, 4200.0)


def hat(rng: np.random.Generator, *, open_: bool = False) -> np.ndarray:
    """The offbeat eighths. Closed keeps time; open marks the bar."""
    decay = 0.22 if open_ else 0.032
    samples = int((decay * 2.2) * SAMPLE_RATE)
    source = synth.noise(samples, rng) * synth.percussive(samples, decay)
    return synth.highpass(source, 6800.0) * (0.5 if open_ else 0.42)


def crash(rng: np.random.Generator, *, decay_s: float = 1.6) -> np.ndarray:
    """Lands on the first beat after a drop, and covers the seam."""
    samples = int(decay_s * 1.4 * SAMPLE_RATE)
    source = synth.noise(samples, rng) * synth.percussive(samples, decay_s, curve=0.7)
    return synth.highpass(source, 3600.0) * 0.5


# --------------------------------------------------------------------------
# pitched voices
# --------------------------------------------------------------------------


def bass(freq: float, length_s: float, rng: np.random.Generator) -> np.ndarray:
    """The rolling offbeat. It plays where the kick is not.

    Placed on the eighths between kicks and cut short, so the bar reads as a
    gallop rather than a drone. The filter closes as the note falls, which is
    what keeps a fast bassline from turning into mud.
    """
    samples = max(64, int(length_s * SAMPLE_RATE))
    tone = synth.saw(freq, samples) * 0.7 + synth.square(freq, samples, duty=0.42) * 0.3
    tone += synth.sine(freq * 0.5, samples) * 0.45  # an octave of weight below
    cutoff = np.linspace(520.0, 190.0, samples)
    shaped = synth.sweep_lowpass(tone, cutoff, resonance=0.35, order=3)
    return shaped * synth.adsr(samples, attack_s=0.003, decay_s=0.05, sustain=0.55, release_s=0.04)


def pluck(freq: float, length_s: float, rng: np.random.Generator) -> np.ndarray:
    """The sixteenth-note arpeggio that carries the top of the track."""
    samples = max(64, int(length_s * SAMPLE_RATE))
    tone = synth.supersaw(freq, samples, voices=5, detune_cents=12.0, rng=rng)
    cutoff = 1400.0 + 5200.0 * synth.percussive(samples, length_s * 0.5)
    shaped = synth.sweep_lowpass(tone, cutoff, resonance=0.45, order=2)
    return shaped * synth.percussive(samples, length_s * 0.7)


def pad(
    freqs: list[float],
    length_s: float,
    rng: np.random.Generator,
    *,
    cutoff: float = 2600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The chord underneath, spread wide. Slow in, slow out, never in the way.

    Every note of the chord gets its own detuned stack; summing them before
    detuning would lock the voices in phase and the chorus would collapse.
    The stacks are panned across each other, which is where a trance pad gets
    its size from — not from an effect added afterwards.
    """
    samples = max(256, int(length_s * SAMPLE_RATE))
    left = np.zeros(samples)
    right = np.zeros(samples)
    for freq in freqs:
        one_l, one_r = synth.supersaw_stereo(freq, samples, voices=7, detune_cents=22.0, rng=rng)
        left += one_l
        right += one_r

    envelope = synth.adsr(
        samples,
        attack_s=min(0.9, length_s * 0.25),
        decay_s=0.3,
        sustain=0.8,
        release_s=min(1.2, length_s * 0.3),
    )
    count = max(1, len(freqs))

    def shape(side: np.ndarray) -> np.ndarray:
        return synth.sweep_lowpass(side / count, cutoff, resonance=0.2, order=2) * envelope

    return shape(left), shape(right)


def lead(freq: float, length_s: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The melody after the drop — the part somebody might hum.

    Narrower than the pad on purpose: a lead spread as wide as its own chords
    stops being a line and becomes more weather.
    """
    samples = max(128, int(length_s * SAMPLE_RATE))
    left, right = synth.supersaw_stereo(
        freq, samples, voices=7, detune_cents=16.0, spread=0.5, rng=rng
    )
    cutoff = 2200.0 + 3800.0 * synth.adsr(
        samples, attack_s=0.02, decay_s=length_s * 0.4, sustain=0.45, release_s=length_s * 0.3
    )
    envelope = synth.adsr(
        samples, attack_s=0.012, decay_s=0.08, sustain=0.75, release_s=length_s * 0.25
    )

    def shape(side: np.ndarray) -> np.ndarray:
        return synth.sweep_lowpass(side, cutoff, resonance=0.4, order=2) * envelope

    return shape(left), shape(right)


def roll(rng: np.random.Generator, *, beats: float, bpm: float) -> np.ndarray:
    """A snare roll that tightens: the bar before something changes.

    Trance announces its own turns. Without a fill, a thirty-two bar drop is
    the same bar thirty-two times and the ear stops following after eight.
    The steps halve as the roll goes on, which is what makes it read as an
    approach rather than a pattern.
    """
    beat_s = 60.0 / bpm
    total = int(beats * beat_s * SAMPLE_RATE)
    out = np.zeros(total)

    at = 0.0
    step = beat_s / 4.0
    while at < beats * beat_s:
        hit = synth.noise(int(0.06 * SAMPLE_RATE), rng)
        hit *= synth.percussive(hit.size, 0.016)
        through = at / max(1e-6, beats * beat_s)
        synth.place(
            out, synth.bandpass(hit, 1100.0, 5200.0) * (0.3 + 0.7 * through), int(at * SAMPLE_RATE)
        )
        at += step
        if through > ROLL_TIGHTENS_AT:
            step = beat_s / 8.0  # twice as fast for the second half
    return out


def reverse_cymbal(rng: np.random.Generator, *, length_s: float = 1.8) -> np.ndarray:
    """Swept backwards into a section change — the oldest trick there is."""
    samples = int(length_s * SAMPLE_RATE)
    source = synth.noise(samples, rng) * synth.percussive(samples, length_s * 0.5)
    return synth.highpass(source, 2200.0)[::-1] * 0.4


def ride(rng: np.random.Generator) -> np.ndarray:
    """Quarter-note shimmer that arrives late in a section and lifts it."""
    samples = int(0.42 * SAMPLE_RATE)
    source = synth.noise(samples, rng) * synth.percussive(samples, 0.14)
    return synth.bandpass(source, 5200.0, 12000.0) * 0.22


def acid(freq: float, length_s: float, rng: np.random.Generator) -> np.ndarray:
    """The 303 idea: one oscillator, a filter that opens hard and shuts fast.

    Where the supersaw lead is wide and soft, this is narrow and mean — the
    resonance is the note. Kept mono and centred because that is where it
    sat on the record it comes from, and spreading it would blur the one
    thing it is for.
    """
    samples = max(128, int(length_s * SAMPLE_RATE))
    tone = synth.saw(freq, samples) * 0.6 + synth.square(freq, samples, duty=0.48) * 0.4
    # A sharp sweep down from well above the note, with a lot of resonance.
    sweep = synth.percussive(samples, length_s * 0.35)
    cutoff = 380.0 + 4200.0 * sweep
    shaped = synth.sweep_lowpass(tone, cutoff, resonance=0.85, order=2)
    return synth.soft_clip(shaped * 1.6, 1.5) * synth.adsr(
        samples, attack_s=0.004, decay_s=0.06, sustain=0.6, release_s=length_s * 0.2
    )


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------


def riser(length_s: float, rng: np.random.Generator) -> np.ndarray:
    """The bars before a drop: noise climbing, pitch climbing, nerves climbing.

    Two things rise together because either alone is unconvincing — the noise
    gives the hiss, the tone gives the destination.
    """
    samples = max(256, int(length_s * SAMPLE_RATE))
    curve = np.linspace(0.0, 1.0, samples) ** 1.7

    hiss = synth.noise(samples, rng)
    hiss = synth.sweep_lowpass(hiss, 300.0 + 9000.0 * curve, resonance=0.5, order=2)

    tone = synth.sine(200.0 * (2.0 ** (curve * 2.6)), samples) * 0.35

    return (hiss * 0.55 + tone) * (curve**1.4)


def downlifter(length_s: float, rng: np.random.Generator) -> np.ndarray:
    """The opposite, for the moment a drop ends and the floor drops out."""
    samples = max(256, int(length_s * SAMPLE_RATE))
    curve = np.linspace(1.0, 0.0, samples) ** 0.8
    hiss = synth.noise(samples, rng)
    hiss = synth.sweep_lowpass(hiss, 200.0 + 7000.0 * curve, order=2)
    return hiss * curve * 0.45


__all__ = [
    "acid",
    "bass",
    "clap",
    "crash",
    "downlifter",
    "hat",
    "hz",
    "kick",
    "lead",
    "pad",
    "pluck",
    "reverse_cymbal",
    "ride",
    "riser",
    "roll",
]
