"""The music generator: does it make trance, and does it make it twice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.music import composer, render, synth, voices


def test_the_same_seed_always_writes_the_same_track():
    """A station has to be rebuildable, and a track worth keeping has to be
    findable again by its number — both need the seed to decide everything."""
    first = composer.render(composer.plan(4242, minutes=1))
    second = composer.render(composer.plan(4242, minutes=1))
    assert np.array_equal(first, second)


def test_different_seeds_write_different_tracks():
    a = composer.render(composer.plan(1, minutes=1))
    b = composer.render(composer.plan(2, minutes=1))
    assert not np.array_equal(a, b[: a.shape[0]] if b.shape[0] >= a.shape[0] else b)


def test_nothing_clips_and_nothing_is_silent():
    audio = composer.render(composer.plan(99, minutes=1))
    assert np.isfinite(audio).all(), "NaN or infinity reached the output"
    assert np.abs(audio).max() <= 1.0, "the mix would clip on the way to disk"
    assert np.abs(audio).max() > 0.5, "the track is far quieter than a master should be"
    assert np.sqrt((audio**2).mean()) > 0.05, "this is silence with edges"


def test_the_beat_is_where_the_plan_says_it_is():
    """Autocorrelating the envelope is how a beat detector works, so if the
    tempo is really in there this finds it — and if the arrangement drifts,
    this is the test that notices."""
    spec = composer.plan(31337, minutes=2)
    mono = composer.render(spec).mean(axis=1)

    start = int(spec.duration_s * 0.45 * synth.SAMPLE_RATE)
    window = mono[start : start + synth.SAMPLE_RATE * 20]
    envelope = np.convolve(np.abs(window), np.ones(256) / 256, mode="same")[::64]
    envelope = envelope - envelope.mean()
    correlation = np.correlate(envelope, envelope, "full")[envelope.size - 1 :]

    per_second = synth.SAMPLE_RATE / 64
    low, high = int(per_second * 0.28), int(per_second * 1.0)  # 60 to 214 BPM
    lag = low + int(np.argmax(correlation[low:high]))
    detected = 60.0 * per_second / lag
    assert abs(detected - spec.bpm) < 2.0, f"heard {detected:.1f} BPM, planned {spec.bpm}"


def test_a_breakdown_is_a_breath_and_not_a_hole():
    """The first version measured nineteen decibels below the drop, which is
    not a breakdown — it is the track stopping."""
    spec = composer.plan(2024, minutes=5, style="uplifting")
    mono = composer.render(spec).mean(axis=1)

    levels = {}
    at = 0.0
    for section in spec.sections:
        end = at + section.bars * spec.bar_s
        chunk = mono[int(at * synth.SAMPLE_RATE) : int(end * synth.SAMPLE_RATE)]
        levels[section.name] = 20 * np.log10(np.sqrt((chunk**2).mean()) + 1e-9)
        at = end

    gap = levels["peak"] - levels["breakdown"]
    assert 4.0 < gap < 16.0, f"the breakdown sits {gap:.1f} dB below the peak"


def test_the_track_gets_where_it_is_going():
    """An intro that arrives at full strength leaves the drop nowhere to go."""
    spec = composer.plan(808, minutes=5, style="uplifting")
    mono = composer.render(spec).mean(axis=1)
    at = 0.0
    levels = {}
    for section in spec.sections:
        end = at + section.bars * spec.bar_s
        chunk = mono[int(at * synth.SAMPLE_RATE) : int(end * synth.SAMPLE_RATE)]
        levels[section.name] = 20 * np.log10(np.sqrt((chunk**2).mean()) + 1e-9)
        at = end
    assert levels["drop"] - levels["intro"] > 1.5, "the intro is already as loud as the drop"


def test_the_length_asked_for_is_the_length_made():
    for minutes in (2.0, 5.0, 9.0):
        spec = composer.plan(7, minutes=minutes)
        assert (
            abs(spec.duration_s / 60.0 - minutes) < 1.0
        ), f"{minutes} became {spec.duration_s / 60}"


def test_every_section_survives_a_short_track():
    """Shrinking used to be carved out of the long sections alone, which left
    a full-length intro in front of an eight-bar drop."""
    spec = composer.plan(11, minutes=2, style="uplifting")
    assert [s.name for s in spec.sections] == [s.name for s in composer.FORM]
    assert all(s.bars >= 8 for s in spec.sections)


def test_the_sidechain_ducks_and_recovers():
    """The pump is the cheapest thing that makes a mix breathe, and a bug
    that left it stuck down would be inaudible in a test that only checked
    for sound."""
    env = synth.sidechain(synth.SAMPLE_RATE, np.array([0, synth.SAMPLE_RATE // 2]), depth=0.7)
    assert env[10] < 0.45, "nothing ducked out of the kick's way"
    assert env[synth.SAMPLE_RATE // 2 - 100] > 0.95, "it never came back up"


def test_the_filter_actually_removes_the_top():
    noisy = synth.noise(synth.SAMPLE_RATE, np.random.default_rng(0))
    quiet = synth.sweep_lowpass(noisy, 400.0, order=3)
    spectrum = np.abs(np.fft.rfft(quiet))
    freqs = np.fft.rfftfreq(quiet.size, 1 / synth.SAMPLE_RATE)
    below = spectrum[(freqs > 100) & (freqs < 300)].mean()
    above = spectrum[(freqs > 4000) & (freqs < 8000)].mean()
    assert above < below * 0.1, "the lowpass let the top through"


def test_a_kick_is_low_and_short():
    kick = voices.kick()
    assert kick.size < synth.SAMPLE_RATE, "a kick that long is a tom"
    spectrum = np.abs(np.fft.rfft(kick))
    freqs = np.fft.rfftfreq(kick.size, 1 / synth.SAMPLE_RATE)
    assert spectrum[(freqs > 30) & (freqs < 120)].sum() > spectrum[freqs > 500].sum()


def _section_audio(spec, audio, name: str):
    at = 0.0
    for section in spec.sections:
        end = at + section.bars * spec.bar_s
        if section.name == name:
            return audio[int(at * synth.SAMPLE_RATE) : int(end * synth.SAMPLE_RATE)]
        at = end
    raise AssertionError(f"no section called {name}")


def test_the_phrase_ends_differently_from_how_it_runs():
    """A thirty-two bar drop used to be one bar played thirty-two times. The
    fill at the end of each eight is what a listener follows, so it has to be
    measurably unlike its neighbours."""
    spec = composer.plan(5150, minutes=5, style="uplifting")
    peak = _section_audio(spec, composer.render(spec).mean(axis=1), "peak")

    bar_samples = int(spec.bar_s * synth.SAMPLE_RATE)
    bars = [peak[i * bar_samples : (i + 1) * bar_samples] for i in range(composer.LONG_PHRASE)]
    spectra = [np.abs(np.fft.rfft(bar))[:4000] for bar in bars if bar.size == bar_samples]

    def likeness(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    steps = [likeness(spectra[i], spectra[i + 1]) for i in range(len(spectra) - 1)]
    assert (
        steps[-1] < min(steps[:-1]) - 0.05
    ), f"the last bar of the phrase is as ordinary as the rest: {steps}"


def test_the_melody_is_a_melody():
    """One note every two bars is a chord spelled slowly, not a tune."""
    spec = composer.plan(616, minutes=5)
    assert len(spec.lead_rhythm) >= 6, "too few notes in a two-bar span to be a line"
    assert len(set(spec.lead_contour)) >= 3, "a contour needs somewhere to go"
    # And the rhythm has to be syncopated: a note on every downbeat is a drill.
    downbeats = {0, 4, 8, 12, 16, 20, 24, 28}
    assert not downbeats.issubset(set(spec.lead_rhythm))


def test_a_contour_can_dip_below_the_root():
    """divmod does the octave bookkeeping; getting it wrong wraps a note a
    seventh the wrong way and the melody turns sour once a phrase."""
    root = 57
    assert composer._scale_note(root, 0) == root
    assert composer._scale_note(root, 7) == root + 12
    assert composer._scale_note(root, -1) == root - 2, "the step below the root is the seventh"
    assert composer._scale_note(root, -7) == root - 12


def test_the_wide_parts_are_wide_and_the_low_parts_are_not():
    """Width belongs above the bass. Panning a kick only weakens the centre,
    and the measurement has to be taken where the wide voices actually live."""
    spec = composer.plan(1234, minutes=5, style="uplifting")
    audio = composer.render(spec)
    peak = _section_audio(spec, audio, "peak")

    def width(chunk, above=None):
        mid = chunk.mean(axis=1)
        side = (chunk[:, 0] - chunk[:, 1]) / 2
        if above:
            mid, side = synth.highpass(mid, above), synth.highpass(side, above)
        return float(np.sqrt((side**2).mean()) / (np.sqrt((mid**2).mean()) + 1e-9))

    assert width(peak, 500) > 0.1, "the pads and the arpeggio are stuck in the middle"
    assert width(peak) < width(peak, 500), "the bottom end is as wide as the top"


def test_nothing_disappears_when_it_is_summed_to_mono():
    """Width by delaying one channel collapses the moment anything sums to
    mono — a phone, a club system, half of YouTube. Panning real voices
    survives it, and this is the test that says which one we built."""
    spec = composer.plan(1234, minutes=3)
    audio = composer.render(spec)
    stereo_rms = float(np.sqrt((audio**2).mean()))
    mono_rms = float(np.sqrt((audio.mean(axis=1) ** 2).mean()))
    lost_db = 20 * np.log10(mono_rms / (stereo_rms + 1e-12) + 1e-12)
    assert lost_db > -1.5, f"{lost_db:.1f} dB vanished when summed to mono"


def test_a_moving_filter_costs_what_a_still_one_does_not():
    """A corner that never moves needs no windows at all, and every note of
    every arpeggio comes through this function."""
    rng = np.random.default_rng(0)
    signal = synth.noise(120_000, rng)
    still = synth.sweep_lowpass(signal, 2600.0)
    moving = synth.sweep_lowpass(signal, np.full(signal.size, 2600.0))
    # Same filter, two routes through the code: the answers must agree.
    assert np.corrcoef(still, moving)[0, 1] > 0.98


def test_names_are_stable_and_readable():
    assert render.name_for(5) == render.name_for(5)
    assert " " in render.name_for(5)


@pytest.mark.parametrize("seed", [3, 400])
def test_a_track_lands_in_the_library_as_a_tagged_mp3(tmp_path: Path, seed: int):
    """The end of the pipeline: the file has to be something the scanner
    accepts and a player can put a name to."""
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is what encodes the mp3")

    made = render.generate(tmp_path, seed=seed, minutes=1.0, bitrate_k=96)
    assert made.path.is_file()
    assert made.path.suffix == ".mp3"
    assert made.relpath.startswith(f"{render.GENERATED_DIR}/")
    assert made.path.stat().st_size > 100_000, "an mp3 that small holds nothing"

    from app.services.audio import library

    assert made.path.suffix in library.SUPPORTED_SUFFIXES
    tags = library.read_tags(made.path)
    assert tags.title == made.title
    assert tags.genre.startswith("Trance")
    assert tags.duration_s > 30


# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------


def _bass_onsets(spec, mono, *, section_index: int = 1, bars: int = 4) -> list[int]:
    """Where the low end attacks inside a bar, in sixteenths, pooled over a
    few bars. A bassline's signature is not what it plays but when."""
    at = sum(x.bars for x in spec.sections[:section_index]) * spec.bar_s
    # Up to the first octave above the root: the last offbeat of a bar is
    # lifted an octave, and a cutoff below it would call that a rest.
    low = synth.sweep_lowpass(mono, 260.0, order=3)
    step = spec.beat_s / 4.0
    hits = []
    for bar in range(bars):
        base = at + bar * spec.bar_s
        energies = []
        for sixteenth in range(16):
            a = int((base + sixteenth * step) * synth.SAMPLE_RATE)
            b = int((base + (sixteenth + 0.5) * step) * synth.SAMPLE_RATE)
            energies.append(float((low[a:b] ** 2).mean()))
        threshold = max(energies) * 0.35
        hits.extend(i for i, e in enumerate(energies) if e > threshold)
    return sorted(set(hits))


def test_every_style_renders_something_playable():
    from app.services.music import styles

    for name in styles.STYLES:
        spec = composer.plan(77, minutes=1.5, style=name)
        audio = composer.render(spec)
        assert np.isfinite(audio).all(), name
        assert 0.5 < np.abs(audio).max() <= 1.0, name
        low, high = styles.STYLES[name].bpm_range
        assert low <= spec.bpm <= high, (name, spec.bpm)


def test_mixed_hands_different_seeds_different_styles():
    """A playlist of ten from the button must not be ten of the same."""
    names = {composer.plan(seed, minutes=1).style.name for seed in range(8)}
    assert len(names) >= 3, names


def test_the_same_seed_gives_the_same_style():
    assert composer.plan(9, minutes=1).style.name == composer.plan(9, minutes=1).style.name


def test_an_unknown_style_is_refused_not_guessed():
    with pytest.raises(ValueError, match="no such style"):
        composer.plan(1, minutes=1, style="hardstyle")


def test_psy_rolls_and_uplifting_bounces():
    """The bassline is the signature: offbeat eighths for uplifting, the
    three sixteenths after every kick for psy. Same chord, different feet —
    and if a refactor ever collapses them into one, this is what notices."""
    psy = composer.plan(4, minutes=2, style="psy")
    up = composer.plan(4, minutes=2, style="uplifting")
    psy_hits = _bass_onsets(psy, composer.render(psy).mean(axis=1))
    up_hits = _bass_onsets(up, composer.render(up).mean(axis=1), section_index=2)

    offbeats = {2, 6, 10, 14}
    assert len(offbeats & set(up_hits)) >= 3, f"uplifting is not on the offbeat: {up_hits}"
    # Psy's bass never sits on the kick and fills the sixteenths between.
    between = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15}
    assert len(set(psy_hits) & between) >= 8, f"psy is not rolling: {psy_hits}"


def test_progressive_is_slower_and_never_drops():
    spec = composer.plan(3, minutes=3, style="progressive")
    assert spec.bpm < 132
    assert "drop" not in {s.name for s in spec.sections}
    # The filter opens across the track instead of arriving all at once.
    openings = [s.filter_open for s in spec.sections if s.kick]
    assert openings[0][0] < 0.5 and any(o[1] >= 1.0 for o in openings)


def test_the_style_reaches_the_file(tmp_path: Path):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is what encodes the mp3")
    made = render.generate(tmp_path, seed=5, minutes=1.0, bitrate_k=96, style="psy")
    assert made.style == "psy"
    from app.services.audio import library

    tags = library.read_tags(made.path)
    assert "psy" in (tags.album or "").lower() or "psy" in (tags.genre or "").lower()
