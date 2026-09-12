"""Arrangement: turning a seed into a track that goes somewhere.

Trance is unusually explicit about its own form, which is what makes it
writable. Sixteen-bar phrases, a chord loop of four, a breakdown that strips
the drums away, a riser, and a drop that puts them back. The shape is not a
limitation to work around — it is the thing listeners are following.

Nothing here is random for its own sake. The seed picks a key, a tempo, a
progression and a set of patterns, and the same seed always builds the same
track — so a station can be regenerated, and a track that turned out well can
be found again by number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.services.music import synth, voices
from app.services.music.styles import MIXED, BassPattern, Section, Style, pick
from app.services.music.synth import SAMPLE_RATE

#: Natural minor. Trance lives here almost exclusively; the major scale reads
#: as celebration rather than the yearning the genre is after.
MINOR = (0, 2, 3, 5, 7, 8, 10)

#: Chord loops as scale degrees. Four bars, one chord each, endlessly.
PROGRESSIONS = (
    (0, 5, 2, 6),  # i  VI III VII — the staple
    (0, 3, 5, 6),  # i  iv VI  VII
    (5, 2, 0, 6),  # VI III i   VII — starts away from home
    (0, 6, 5, 2),  # i  VII VI III
)

#: Lead rhythms: which sixteenths of a two-bar span carry a note. Trance
#: melodies are syncopated — a note on every beat reads as an exercise — so
#: none of these sits squarely on all four.
LEAD_RHYTHMS = (
    (0, 6, 8, 12, 16, 22, 24, 28),
    (0, 4, 6, 8, 16, 20, 22, 24),
    (0, 8, 12, 14, 16, 24, 28, 30),
    (0, 3, 6, 8, 12, 16, 19, 26),
)

#: Contours: scale steps away from the chord's root, one per onset above. A
#: melody is a shape before it is a set of pitches, and reusing the shape over
#: a changing chord is what makes eight bars sound composed rather than
#: sampled from a scale.
CONTOURS = (
    (0, 2, 4, 2, 7, 4, 2, 0),
    (0, 4, 2, 7, 4, 2, 0, -3),
    (7, 4, 2, 0, 2, 4, 7, 9),
    (0, 0, 2, 4, 4, 2, 0, -1),
)

#: Sixteenth-note arpeggio shapes, as indices into the chord's notes.
#: Counting units. Named because the arrangement reads as music this way
#: and as arithmetic otherwise.
BEATS_PER_BAR = 4
EIGHTHS_PER_BAR = 8
STEPS_PER_BAR = 16
PHRASE_BARS = 4
#: The unit a listener actually follows. Something has to change at the end of
#: each of these or a thirty-two bar drop is one bar played thirty-two times.
LONG_PHRASE = 8
#: Shorter than this and there is no room for a breakdown.
MIN_BARS = 48
#: Sections at least this long are the ones a longer track grows into.
LONG_SECTION = 24

ARPS = (
    (0, 1, 2, 3, 2, 1, 2, 1),
    (0, 2, 1, 3, 0, 2, 1, 3),
    (3, 2, 1, 0, 1, 2, 3, 2),
    (0, 1, 0, 2, 0, 1, 3, 2),
)


#: The default journey, kept under its old name for anything that asks.
FORM = pick("uplifting", 0).form

#: With the drums gone, the chords and the melody are the only things left —
#: at their usual mix level a breakdown measured nineteen decibels below the
#: drop, which is a hole in the track rather than a breath in it.
BREAKDOWN_LIFT = 2.3


@dataclass
class Plan:
    """Everything decided before a single sample is written."""

    seed: int
    bpm: float
    root_midi: int
    progression: tuple[int, ...]
    arp_shape: tuple[int, ...]
    lead_rhythm: tuple[int, ...] = LEAD_RHYTHMS[0]
    lead_contour: tuple[int, ...] = CONTOURS[0]
    sections: list[Section] = field(default_factory=list)
    style: Style = field(default_factory=lambda: pick("uplifting", 0))

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_s(self) -> float:
        return self.beat_s * 4.0

    @property
    def total_bars(self) -> int:
        return sum(section.bars for section in self.sections)

    @property
    def duration_s(self) -> float:
        return self.total_bars * self.bar_s

    @property
    def key_name(self) -> str:
        names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        return f"{names[self.root_midi % 12]} minor"


def plan(seed: int, *, minutes: float = 5.0, style: str = MIXED) -> Plan:
    """Decide the track. Deterministic in the seed, always."""
    chosen = pick(style, seed)
    rng = np.random.default_rng(seed)
    low, high = chosen.bpm_range
    bpm = float(rng.integers(low, high + 1))
    root = int(rng.integers(45, 57))  # A2 to A3: low enough to sit under a mix

    sections = list(chosen.form)
    target_bars = max(MIN_BARS, round(minutes * 60.0 / (60.0 / bpm * BEATS_PER_BAR)))
    base_bars = sum(section.bars for section in sections)
    if target_bars < base_bars:
        # Proportional, not carved out of the long sections: a short track
        # with a full-length intro and an eight-bar drop is lopsided rather
        # than short. Eight bars is the floor — below that a section reads as
        # a stumble.
        scale = target_bars / base_bars
        sections = [
            Section(**{**s.__dict__, "bars": max(8, round(s.bars * scale / 8) * 8)})
            for s in sections
        ]
    elif target_bars > base_bars:
        # Lengthen where a track is meant to be long: the drops and the
        # breakdown. Stretching the intro would only delay the track.
        extra = target_bars - base_bars
        grown = []
        # The long sections are the ones without a filter sweep written into
        # them: a build that is meant to open over sixteen bars should not
        # be told to take forty.
        stretchy = [x for x in sections if x.bars >= LONG_SECTION] or sections[1:-1]
        for section in sections:
            if section in stretchy:
                share = int(extra / len(stretchy)) // 8 * 8
                grown.append(Section(**{**section.__dict__, "bars": section.bars + share}))
            else:
                grown.append(section)
        sections = grown

    return Plan(
        seed=seed,
        bpm=bpm,
        root_midi=root,
        progression=PROGRESSIONS[int(rng.integers(0, len(PROGRESSIONS)))],
        arp_shape=ARPS[int(rng.integers(0, len(ARPS)))],
        lead_rhythm=LEAD_RHYTHMS[int(rng.integers(0, len(LEAD_RHYTHMS)))],
        lead_contour=CONTOURS[int(rng.integers(0, len(CONTOURS)))],
        sections=sections,
        style=chosen,
    )


def _scale_note(root_midi: int, degree: int) -> int:
    """A note this many scale steps above the root, in either direction.

    divmod does the octave bookkeeping, negatives included, which is what lets
    a contour dip below the root without wrapping to the wrong note.
    """
    octave, index = divmod(degree, len(MINOR))
    return root_midi + MINOR[index] + 12 * octave


def _triad(root_midi: int, degree: int) -> list[int]:
    """A three-note chord built on a degree of the minor scale."""
    return [
        root_midi + MINOR[(degree + step) % 7] + 12 * ((degree + step) // 7) for step in (0, 2, 4)
    ]


@dataclass
class _Canvas:
    """The stems a track is built from, kept apart until the mix.

    Separate buses are not tidiness: the kick has to stay out of the ducking
    it causes, and the reverb belongs on the melodic parts only. Mixing as we
    go would make both impossible.
    """

    #: Kick and bass stay in the middle: low frequencies carry no usable
    #: direction, and anything panned down there only weakens the centre.
    drums: np.ndarray
    low: np.ndarray
    mid_l: np.ndarray
    mid_r: np.ndarray
    high_l: np.ndarray
    high_r: np.ndarray
    effects: np.ndarray
    kick_hits: list[int] = field(default_factory=list)

    @classmethod
    def blank(cls, samples: int) -> _Canvas:
        return cls(*(np.zeros(samples) for _ in range(7)))


@dataclass(frozen=True)
class _Kit:
    """Voices that never vary, rendered once instead of several hundred times."""

    kick: np.ndarray
    clap: np.ndarray
    closed_hat: np.ndarray
    open_hat: np.ndarray
    ride: np.ndarray

    @classmethod
    def build(cls, rng: np.random.Generator, *, punch: float = 1.0) -> _Kit:
        return cls(
            voices.kick(punch=punch),
            voices.clap(rng),
            voices.hat(rng),
            voices.hat(rng, open_=True),
            voices.ride(rng),
        )


def _lay_hats(
    canvas: _Canvas,
    kit: _Kit,
    *,
    at_bar: float,
    beat: float,
    bar_no: int,
    opening: float,
    level: float,
    riding: bool,
    sr: int,
) -> None:
    """Offbeat eighths, then sixteenths, then a ride — the section grows
    without the arrangement changing underneath it."""
    in_phrase = bar_no % LONG_PHRASE

    for eighth in range(1, EIGHTHS_PER_BAR, 2):
        marks_phrase = eighth == EIGHTHS_PER_BAR - 1 and bar_no % PHRASE_BARS == PHRASE_BARS - 1
        sound = kit.open_hat if marks_phrase else kit.closed_hat
        synth.place(canvas.drums, sound * opening * level, int((at_bar + eighth * beat / 2.0) * sr))

    # Sixteenths arrive halfway through a phrase and double the drive without
    # adding anything new to listen to.
    if in_phrase >= LONG_PHRASE // 2:
        for step in range(1, STEPS_PER_BAR, 2):
            synth.place(
                canvas.drums,
                kit.closed_hat * 0.4 * opening * level,
                int((at_bar + step * beat / 4.0) * sr),
            )

    if bar_no // LONG_PHRASE >= 1 and riding:
        for index in range(BEATS_PER_BAR):
            synth.place(canvas.drums, kit.ride * opening * level, int((at_bar + index * beat) * sr))


def _lay_drums(
    canvas: _Canvas,
    kit: _Kit,
    section: Section,
    rng: np.random.Generator,
    *,
    at_bar: float,
    beat: float,
    bar_no: int,
    opening: float,
    level: float,
    bpm: float,
    sr: int,
) -> None:
    """The kit, and the small changes that keep eight bars from being one bar.

    Everything that varies here varies on the phrase, never on the bar: a
    pattern that changes constantly has no pattern to depart from, and the
    departure is the point.
    """
    is_fill = bar_no % LONG_PHRASE == LONG_PHRASE - 1

    if section.kick:
        for index in range(BEATS_PER_BAR):
            # The fill takes the last beat; a kick under a roll muddies both.
            if is_fill and index == BEATS_PER_BAR - 1:
                continue
            at = int((at_bar + index * beat) * sr)
            canvas.kick_hits.append(at)
            synth.place(canvas.drums, kit.kick * level, at)

    if section.clap:
        backbeats = (1, BEATS_PER_BAR - 1)
        for index in backbeats:
            if is_fill and index == backbeats[-1]:
                continue
            synth.place(canvas.drums, kit.clap * 0.55 * level, int((at_bar + index * beat) * sr))

    if section.hats:
        _lay_hats(
            canvas,
            kit,
            at_bar=at_bar,
            beat=beat,
            bar_no=bar_no,
            opening=opening,
            level=level,
            riding=section.kick,
            sr=sr,
        )

    if is_fill and (section.kick or section.hats):
        fill = voices.roll(rng, beats=1.0, bpm=bpm)
        synth.place(
            canvas.drums,
            fill * 0.5 * level,
            int((at_bar + (BEATS_PER_BAR - 1) * beat) * sr),
        )


def _lay_bass(
    canvas: _Canvas,
    spec: Plan,
    chord: list[int],
    rng: np.random.Generator,
    *,
    at_bar: float,
    beat: float,
    bar_no: int,
    level: float,
    sr: int,
) -> None:
    """The bassline is the signature of the sub-genre, so this is where the
    styles part company: the same chord, three different feet."""
    root = chord[0] - 12
    is_fill = bar_no % LONG_PHRASE == LONG_PHRASE - 1
    pattern = spec.style.bass

    if pattern is BassPattern.ROLLING:
        # Kick on the beat, bass on the three sixteenths after it. The kick
        # owns the downbeat; the bass fills the space it leaves and never
        # lands on top of it. This is the whole of psytrance's engine.
        for step in range(STEPS_PER_BAR):
            if step % (STEPS_PER_BAR // BEATS_PER_BAR) == 0:
                continue
            per_beat = STEPS_PER_BAR // BEATS_PER_BAR
            last_of_group = step % per_beat == per_beat - 1
            pitch = root + 12 if (last_of_group and bar_no % 2 == 1) else root
            note = voices.bass(voices.hz(pitch), beat * 0.19, rng)
            synth.place(canvas.low, note * 0.95 * level, int((at_bar + step * beat / 4.0) * sr))
        return

    if pattern is BassPattern.PULSE:
        # With the kick, long and soft: a floor rather than a gallop. The
        # sidechain gives it the pulse; it does not need one of its own.
        for index in range(BEATS_PER_BAR):
            ends_phrase = index == BEATS_PER_BAR - 1 and bar_no % PHRASE_BARS == PHRASE_BARS - 1
            pitch = root + 7 if ends_phrase else root
            note = voices.bass(voices.hz(pitch), beat * 0.95, rng)
            synth.place(canvas.low, note * 0.7 * level, int((at_bar + index * beat) * sr))
        return

    # Offbeat: the note lands between kicks, never on one. The root on every
    # offbeat is correct and lifeless, so the last one of the bar lifts an
    # octave and the fill runs sixteenths up to meet the next phrase.
    for eighth in range(1, EIGHTHS_PER_BAR, 2):
        pitch = root + 12 if eighth == EIGHTHS_PER_BAR - 1 else root
        if is_fill and eighth >= EIGHTHS_PER_BAR - 2:
            for half in range(2):
                note = voices.bass(voices.hz(pitch + half * 3), beat * 0.2, rng)
                at = at_bar + eighth * beat / 2.0 + half * beat / 4.0
                synth.place(canvas.low, note * 0.85 * level, int(at * sr))
            continue
        note = voices.bass(voices.hz(pitch), beat * 0.42, rng)
        synth.place(canvas.low, note * 0.9 * level, int((at_bar + eighth * beat / 2.0) * sr))


def _lay_arp(
    canvas: _Canvas,
    spec: Plan,
    chord: list[int],
    rng: np.random.Generator,
    *,
    at_bar: float,
    beat: float,
    opening: float,
    level: float,
    sr: int,
) -> None:
    notes = [chord[0], chord[1], chord[2], chord[0] + 12]
    half = STEPS_PER_BAR // 2
    for step in range(STEPS_PER_BAR):
        which = spec.arp_shape[step % len(spec.arp_shape)]
        octave = 12 if step % half >= half // 2 else 0
        voice = voices.pluck(
            voices.hz(notes[which] + 12 + octave), beat * spec.style.arp_length, rng
        )
        at = int((at_bar + step * beat / 4.0) * sr)
        # Alternate sides. A sixteenth-note arpeggio bouncing across the field
        # is half of what makes a trance mix feel wide, and it costs nothing.
        near, far = (0.92, 0.38) if step % 2 == 0 else (0.38, 0.92)
        gain = 0.42 * opening * level
        synth.place(canvas.mid_l, voice * gain * near, at)
        synth.place(canvas.mid_r, voice * gain * far, at)


def _lay_chords(
    canvas: _Canvas,
    spec: Plan,
    section: Section,
    voicing: tuple[list[int], int],
    rng: np.random.Generator,
    *,
    at_bar: float,
    bar: float,
    bar_no: int,
    opening: float,
    level: float,
    sr: int,
) -> None:
    chord, chord_degree = voicing
    carrying = BREAKDOWN_LIFT if not section.kick else 1.0

    if section.pad and bar_no % 2 == 0:
        gain = 0.45 * opening * level * carrying
        left, right = voices.pad(
            [voices.hz(n) for n in chord], bar * 2.0, rng, cutoff=spec.style.pad_cutoff
        )
        synth.place_stereo(
            canvas.mid_l, canvas.mid_r, (left * gain, right * gain), int(at_bar * sr)
        )
        if carrying > 1.0:
            # Something has to hold the bottom while the kick is away.
            below_l, below_r = voices.pad([voices.hz(chord[0] - 12)], bar * 2.0, rng)
            synth.place(canvas.low, (below_l + below_r) * 0.15 * level, int(at_bar * sr))

    if section.lead and bar_no % 2 == 0:
        _lay_lead(
            canvas,
            spec,
            chord_degree,
            rng,
            at_bar=at_bar,
            beat=bar / BEATS_PER_BAR,
            level=level * carrying,
            answering=(bar_no % LONG_PHRASE) >= LONG_PHRASE - 2,
            sr=sr,
        )


def _lay_lead(
    canvas: _Canvas,
    spec: Plan,
    chord_degree: int,
    rng: np.random.Generator,
    *,
    at_bar: float,
    beat: float,
    level: float,
    answering: bool,
    sr: int,
) -> None:
    """The part somebody might hum, across two bars.

    One note every two bars is a chord spelled slowly, not a tune. The rhythm
    and the contour are fixed for the whole track and read against whatever
    chord is underneath, which is how eight bars come out sounding composed
    instead of drawn from a scale at random.
    """
    onsets = spec.lead_rhythm
    span = STEPS_PER_BAR * 2
    for index, step in enumerate(onsets):
        offset = spec.lead_contour[index % len(spec.lead_contour)]
        # The last note of a phrase falls home, so the phrase has an ending
        # rather than simply stopping.
        if answering and index == len(onsets) - 1:
            offset = 0
        note = _scale_note(spec.root_midi, chord_degree + offset) + 24
        until = onsets[index + 1] if index + 1 < len(onsets) else span
        length = max(0.12, (until - step) * beat / 4.0)
        at = int((at_bar + step * beat / 4.0) * sr)
        gain = 0.34 * level * spec.style.lead_gain
        if spec.style.acid_lead:
            # Narrow and centred, on purpose: the resonance is the note.
            voice = voices.acid(voices.hz(note - 12), min(length, beat), rng)
            synth.place(canvas.high_l, voice * gain, at)
            synth.place(canvas.high_r, voice * gain, at)
            continue
        left, right = voices.lead(voices.hz(note), min(length, beat * 2.0), rng)
        synth.place_stereo(canvas.high_l, canvas.high_r, (left * gain, right * gain), at)


def _mixdown(canvas: _Canvas, rng: np.random.Generator, *, beat: float) -> np.ndarray:
    """Glue, space and the pump — everything that happens after the notes.

    The ducking is the one step that cannot be skipped: trance without the
    kick carving a hole for itself is a wall of sound with no pulse in it.
    """
    duck = synth.sidechain(
        canvas.drums.size, np.array(canvas.kick_hits, dtype=np.int64), depth=0.62
    )
    canvas.low *= duck
    canvas.mid_l *= duck
    canvas.mid_r *= duck
    lightly = duck * 0.85 + 0.15
    canvas.high_l *= lightly
    canvas.high_r *= lightly

    # Ping-ponged: each echo answers on the other side, which widens the tail
    # instead of thickening the middle.
    mid_l, mid_r = canvas.mid_l.copy(), canvas.mid_r.copy()
    canvas.mid_l += synth.delay_line(mid_r, time_s=beat * 0.75, feedback=0.3, repeats=5) * 0.35
    canvas.mid_r += synth.delay_line(mid_l, time_s=beat * 0.75, feedback=0.3, repeats=5) * 0.35
    high_l, high_r = canvas.high_l.copy(), canvas.high_r.copy()
    canvas.high_l += synth.delay_line(high_r, time_s=beat * 1.5, feedback=0.34, repeats=4) * 0.4
    canvas.high_r += synth.delay_line(high_l, time_s=beat * 1.5, feedback=0.34, repeats=4) * 0.4

    wet_l = synth.reverb(canvas.mid_l * 0.35 + canvas.high_l * 0.45, rng=rng, seconds=2.2)
    wet_r = synth.reverb(canvas.mid_r * 0.35 + canvas.high_r * 0.45, rng=rng, seconds=2.2)

    centre = canvas.drums * 0.9 + canvas.low + canvas.effects
    left = synth.soft_clip((centre + canvas.mid_l + canvas.high_l + wet_l * 0.5) * 0.62, 1.3)
    right = synth.soft_clip((centre + canvas.mid_r + canvas.high_r + wet_r * 0.5) * 0.62, 1.3)

    # One gain for both sides: normalising each on its own would shift the
    # image whenever the two happened to peak differently.
    loudest = max(float(np.abs(left).max()), float(np.abs(right).max()), synth.SILENCE)
    gain = 0.89 / loudest
    return synth.stereo(left * gain, right * gain)


def render(spec: Plan) -> np.ndarray:
    """Build the whole track and return it as stereo float samples."""
    rng = np.random.default_rng(spec.seed)
    sr = SAMPLE_RATE
    beat, bar = spec.beat_s, spec.bar_s
    canvas = _Canvas.blank(int(spec.duration_s * sr) + sr)  # a second for the tails
    kit = _Kit.build(rng, punch=spec.style.kick_punch)

    bar_index = 0
    for section in spec.sections:
        section_start = bar_index * bar
        for local_bar in range(section.bars):
            at_bar = section_start + local_bar * bar
            chord_degree = spec.progression[bar_index % len(spec.progression)]
            chord = _triad(spec.root_midi, chord_degree)
            through = local_bar / max(1, section.bars - 1) if section.bars > 1 else 1.0
            opens_from, opens_to = section.filter_open
            opening = opens_from + through * (opens_to - opens_from)

            _lay_drums(
                canvas,
                kit,
                section,
                rng,
                at_bar=at_bar,
                beat=beat,
                bar_no=local_bar,
                opening=opening,
                level=section.level,
                bpm=spec.bpm,
                sr=sr,
            )
            if section.bass:
                _lay_bass(
                    canvas,
                    spec,
                    chord,
                    rng,
                    at_bar=at_bar,
                    beat=beat,
                    bar_no=local_bar,
                    level=section.level,
                    sr=sr,
                )
            if section.arp:
                _lay_arp(
                    canvas,
                    spec,
                    chord,
                    rng,
                    at_bar=at_bar,
                    beat=beat,
                    opening=opening,
                    level=section.level,
                    sr=sr,
                )
            _lay_chords(
                canvas,
                spec,
                section,
                (chord, chord_degree),
                rng,
                at_bar=at_bar,
                bar=bar,
                bar_no=local_bar,
                opening=opening,
                level=section.level,
                sr=sr,
            )
            bar_index += 1

        ends_at = section_start + section.bars * bar
        if section.riser:
            length = min(section.bars, 8) * bar
            synth.place(
                canvas.effects, voices.riser(length, rng) * 0.3, int((ends_at - length) * sr)
            )
            synth.place(canvas.effects, voices.crash(rng) * 0.4, int(ends_at * sr))
        elif not section.kick:
            # Coming out of a breakdown the drums return on their own, which
            # lands flat. A cymbal swept backwards into the boundary is the
            # oldest way of saying "here it comes" and still the clearest.
            sweep = voices.reverse_cymbal(rng, length_s=bar)
            synth.place(canvas.effects, sweep, int((ends_at - bar) * sr))

    return _mixdown(canvas, rng, beat=beat)


__all__ = ["ARPS", "FORM", "MINOR", "PROGRESSIONS", "Plan", "Section", "plan", "render"]
