"""The xfade chain: offsets, clamping, and the duration that results."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from app.core.config import Config
from app.services.visual.animator import Clip
from app.services.visual.blockbuilder import (
    build_filtergraph,
    plan_transitions,
    total_duration,
)
from app.services.visual.encoder import EncodeProfile


def clip(duration: float, name: str = "plasma") -> Clip:
    return Clip(
        path=Path(f"/tmp/{name}.mkv"),
        duration_s=duration,
        generator=name,
        palette="deep_ocean",
        seed=1,
        loops=True,
        frames=int(duration * 30),
        render_size=(640, 360),
    )


PROFILE = EncodeProfile._build(Config(), "libx264")


def offsets_in(graph: str) -> list[float]:
    return [float(m) for m in re.findall(r"offset=([0-9.]+)", graph)]


def durations_in(graph: str) -> list[float]:
    return [float(m) for m in re.findall(r"xfade=transition=\w+:duration=([0-9.]+)", graph)]


def test_two_clips_overlap_by_exactly_the_transition():
    clips = [clip(30.0), clip(30.0)]
    graph, total = build_filtergraph(clips, [4.0], ["fade"], PROFILE, 1.0)
    # The second clip starts 4s before the first one ends.
    assert offsets_in(graph) == [26.0]
    assert total == pytest.approx(56.0)


def test_offsets_accumulate_across_a_chain():
    clips = [clip(20.0), clip(30.0), clip(25.0), clip(40.0)]
    transitions = [3.0, 4.0, 5.0]
    graph, total = build_filtergraph(clips, transitions, ["fade"] * 3, PROFILE, 1.0)

    # Each offset is where the running output has reached, minus the overlap.
    assert offsets_in(graph) == pytest.approx([17.0, 43.0, 63.0])
    assert total == pytest.approx(20 + 30 + 25 + 40 - 3 - 4 - 5)
    assert total == pytest.approx(total_duration(clips, transitions))


def test_every_input_is_normalised_before_being_faded():
    """xfade refuses mismatched frame rates or timebases."""
    clips = [clip(10.0), clip(10.0)]
    graph, _ = build_filtergraph(clips, [2.0], ["fade"], PROFILE, 1.0)
    for index in range(2):
        assert f"[{index}:v]fps=30" in graph
        assert "setpts=PTS-STARTPTS" in graph
        assert "settb=AVTB" in graph


def test_edge_fades_are_applied_at_both_ends():
    clips = [clip(30.0), clip(30.0)]
    graph, total = build_filtergraph(clips, [4.0], ["fade"], PROFILE, 1.0)
    assert "fade=t=in:st=0:d=1.000" in graph
    assert f"fade=t=out:st={total - 1:.3f}" in graph


def test_a_single_clip_still_produces_a_valid_graph():
    graph, total = build_filtergraph([clip(30.0)], [], [], PROFILE, 1.0)
    assert "xfade" not in graph
    assert "[out]" in graph
    assert total == pytest.approx(30.0)


def test_edge_fade_cannot_swallow_a_short_block():
    _, total = build_filtergraph([clip(2.0)], [], [], PROFILE, 10.0)
    assert total == pytest.approx(2.0)


def test_the_graph_always_ends_on_the_out_label():
    for count in (1, 2, 5):
        clips = [clip(10.0) for _ in range(count)]
        transitions = [2.0] * (count - 1)
        graph, _ = build_filtergraph(clips, transitions, ["fade"] * (count - 1), PROFILE, 0.5)
        assert graph.rstrip().endswith("[out]")


def test_offsets_never_go_negative():
    """A transition longer than the first clip must not produce offset=-x."""
    clips = [clip(2.0), clip(30.0)]
    graph, _ = build_filtergraph(clips, [5.0], ["fade"], PROFILE, 0.5)
    assert all(offset >= 0 for offset in offsets_in(graph))


# --- transition planning -------------------------------------------------


def test_transitions_are_clamped_to_the_shorter_neighbour():
    """Otherwise a 20s clip next to a 5s one is almost entirely crossfade."""
    cfg = Config.model_validate(
        {"visual": {"block": {"transition_min_s": 5.0, "transition_max_s": 5.0}}}
    )
    durations, _ = plan_transitions([clip(30.0), clip(6.0)], cfg, random.Random(1))
    assert durations[0] <= 6.0 * 0.4 + 1e-9


def test_transition_names_come_from_the_configured_set():
    cfg = Config.model_validate({"visual": {"block": {"transitions": ["radial", "pixelize"]}}})
    _, names = plan_transitions([clip(30.0)] * 4, cfg, random.Random(7))
    assert set(names) <= {"radial", "pixelize"}
    assert len(names) == 3


def test_planning_is_reproducible_for_a_seed():
    cfg = Config()
    clips = [clip(25.0) for _ in range(5)]
    first = plan_transitions(clips, cfg, random.Random(42))
    second = plan_transitions(clips, cfg, random.Random(42))
    assert first == second


def test_the_join_is_given_time_in_proportion_to_the_block():
    """A fixed timeout threw away an hour and fifty minutes of rendered clips
    on an ARM NAS, because joining a ten-minute block there takes longer than
    fifteen minutes. The limit has to scale with the work, and it must never
    be *less* than the configured floor either."""
    from app.services.visual.blockbuilder import JOIN_SECONDS_PER_VIDEO_SECOND

    floor = 900.0
    ten_minutes = 620.0
    assert (
        max(floor, ten_minutes * JOIN_SECONDS_PER_VIDEO_SECOND) > 3600
    ), "a ten-minute block on a slow NAS needs more than an hour"
    assert (
        max(floor, 10.0 * JOIN_SECONDS_PER_VIDEO_SECOND) == floor
    ), "a short block keeps the configured floor"
