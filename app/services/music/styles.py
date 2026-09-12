"""The sub-genres: what makes a psytrance track not an uplifting one.

Tempo is the least of it. The bassline is the signature — offbeat eighths in
uplifting, the rolling sixteenths that psytrance is built on, the softer
on-the-beat pulse of progressive — and the form is the journey: uplifting
climbs to a drop, progressive never quite arrives, psy barely stops. A
station that only varied the tempo between tracks would still be playing
the same track at different speeds, and the ear knows it after three.

Each style is a complete description; the composer asks it questions and
never checks its name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Section:
    """A stretch of bars and what is allowed to play during it."""

    name: str
    bars: int
    kick: bool = True
    bass: bool = True
    hats: bool = True
    clap: bool = True
    arp: bool = True
    pad: bool = True
    lead: bool = False
    riser: bool = False
    #: Opens over the section, so a build can be heard arriving.
    filter_open: tuple[float, float] = (1.0, 1.0)
    #: How loud the section is meant to sit. Without this every part of the
    #: track measured within half a decibel of every other: an intro that
    #: arrives at full strength leaves the drop with nothing to arrive from.
    level: float = 1.0


class BassPattern(str, Enum):
    #: The note lands between the kicks. Uplifting, tech, most of the genre.
    OFFBEAT = "offbeat"
    #: Kick on the beat, bass on the three sixteenths after it — the roll
    #: that psytrance is built around. "K-b-b-b, K-b-b-b".
    ROLLING = "rolling"
    #: With the kick, softer and longer. Progressive, where the bass is a
    #: floor rather than a gallop.
    PULSE = "pulse"


@dataclass(frozen=True)
class Style:
    name: str
    #: One line for the dashboard, so choosing is not a guess.
    blurb: str
    bpm_range: tuple[int, int]
    form: tuple[Section, ...]
    bass: BassPattern = BassPattern.OFFBEAT
    #: How hard the kick's click is. Progressive wants it round; psy wants it
    #: to cut through a bassline that never stops.
    kick_punch: float = 1.0
    #: Whether the lead is a melody or, in tech, mostly absent.
    lead_gain: float = 1.0
    #: Arpeggio note length as a fraction of a beat: shorter is pluckier.
    arp_length: float = 0.3
    #: Where the pad's lowpass sits at rest. Lower is darker.
    pad_cutoff: float = 2600.0
    #: Whether the melody gets the acid treatment — a resonant filter that
    #: opens across each note — instead of the wide supersaw.
    acid_lead: bool = False


# Forms. Bars are proportions; the composer scales them to the length asked.

UPLIFTING_FORM = (
    Section(
        "intro",
        16,
        bass=False,
        clap=False,
        arp=False,
        pad=False,
        filter_open=(0.25, 0.6),
        level=0.62,
    ),
    Section("build", 16, arp=False, lead=False, riser=True, filter_open=(0.45, 1.0), level=0.82),
    Section("drop", 32, lead=True, level=1.0),
    Section("breakdown", 24, kick=False, bass=False, clap=False, hats=False, lead=True, level=1.0),
    Section("lift", 16, clap=False, riser=True, filter_open=(0.3, 1.0), level=0.9),
    Section("peak", 32, lead=True, level=1.0),
    Section("outro", 16, arp=False, lead=False, clap=False, filter_open=(1.0, 0.35), level=0.7),
)

#: Progressive never arrives. The filter opens across the whole track and
#: there is no drop, only a groove that deepens; the one breakdown is short
#: and the lead is sparse.
PROGRESSIVE_FORM = (
    Section(
        "intro",
        16,
        bass=False,
        clap=False,
        arp=False,
        lead=False,
        filter_open=(0.2, 0.45),
        level=0.6,
    ),
    Section("groove", 32, clap=False, lead=False, filter_open=(0.45, 0.75), level=0.8),
    Section("deepen", 32, lead=True, filter_open=(0.75, 1.0), level=0.92),
    Section("breath", 16, kick=False, bass=False, clap=False, hats=False, lead=True, level=0.95),
    Section("return", 40, lead=True, filter_open=(0.7, 1.0), level=1.0),
    Section("outro", 16, arp=False, lead=False, clap=False, filter_open=(1.0, 0.3), level=0.65),
)

#: Psytrance barely stops. Long rolling sections, a breakdown that is a
#: pause rather than a rest, no clap — the roll is the rhythm.
PSY_FORM = (
    Section(
        "intro", 16, clap=False, arp=False, pad=False, lead=False, filter_open=(0.3, 0.8), level=0.7
    ),
    Section("roll", 40, clap=False, lead=True, level=1.0),
    Section(
        "pause",
        8,
        kick=False,
        bass=False,
        clap=False,
        hats=False,
        lead=True,
        riser=True,
        level=0.95,
    ),
    Section("roll again", 48, clap=False, lead=True, level=1.0),
    Section("outro", 16, clap=False, arp=False, lead=False, filter_open=(1.0, 0.4), level=0.7),
)

#: Tech-trance: the groove is the point. Melody kept to a suggestion, the
#: arpeggio and the filter do the talking, one short break.
TECH_FORM = (
    Section(
        "intro",
        16,
        bass=False,
        arp=False,
        pad=False,
        lead=False,
        filter_open=(0.3, 0.6),
        level=0.65,
    ),
    Section("groove", 40, lead=False, filter_open=(0.5, 1.0), level=0.95),
    Section(
        "break",
        16,
        kick=False,
        bass=False,
        clap=False,
        hats=False,
        lead=True,
        riser=True,
        level=0.9,
    ),
    Section("drive", 48, lead=True, level=1.0),
    Section("outro", 16, arp=False, lead=False, filter_open=(1.0, 0.35), level=0.7),
)


STYLES: dict[str, Style] = {
    "uplifting": Style(
        name="uplifting",
        blurb="the big one — builds, breaks, drops, a melody you could hum",
        bpm_range=(136, 142),
        form=UPLIFTING_FORM,
    ),
    "progressive": Style(
        name="progressive",
        blurb="slower and deeper — no drop, the groove just keeps opening",
        bpm_range=(126, 131),
        form=PROGRESSIVE_FORM,
        bass=BassPattern.PULSE,
        kick_punch=0.55,
        lead_gain=0.6,
        arp_length=0.45,
        pad_cutoff=1800.0,
    ),
    "psy": Style(
        name="psy",
        blurb="fast and relentless — the rolling bassline, an acid lead, hardly a pause",
        bpm_range=(142, 148),
        form=PSY_FORM,
        bass=BassPattern.ROLLING,
        kick_punch=1.3,
        arp_length=0.2,
        pad_cutoff=3200.0,
        acid_lead=True,
    ),
    "tech": Style(
        name="tech",
        blurb="driving and stripped back — filter and groove, melody kept to a hint",
        bpm_range=(138, 142),
        form=TECH_FORM,
        kick_punch=1.15,
        lead_gain=0.4,
        arp_length=0.25,
        pad_cutoff=2200.0,
    ),
}

#: What the dashboard offers when nobody chooses: every style in turn, so a
#: playlist of ten is not ten of the same.
MIXED = "mixed"


def pick(name: str, seed: int) -> Style:
    """The style asked for, or one chosen by the seed when the answer is 'mixed'."""
    if name and name != MIXED:
        try:
            return STYLES[name]
        except KeyError:
            raise ValueError(f"no such style: {name!r}; one of {', '.join(STYLES)}") from None
    names = sorted(STYLES)
    return STYLES[names[seed % len(names)]]


__all__ = ["MIXED", "STYLES", "BassPattern", "Section", "Style", "pick"]
