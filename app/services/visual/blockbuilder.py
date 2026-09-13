"""Join clips into one ready-to-stream block, crossfades baked in.

This is the step that makes the whole architecture work. The streamer copies
video bytes and composites nothing, so every transition a viewer will ever see
has to exist inside a file before the broadcast starts. Here is where they are
burned in.

Two details are load-bearing:

* **The block starts and ends with a fade from and to black.** Blocks are
  concatenated live, back to back, with no crossfade possible between them.
  Black-to-black joins are the one kind of cut that looks deliberate.
* **The encode uses the shared profile, unmodified.** See
  :mod:`app.services.visual.encoder` for why any deviation is fatal rather
  than merely untidy.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Config
from app.core.logging import get_logger
from app.core.proc import CommandError, RunOptions, run
from app.services.visual.animator import Clip
from app.services.visual.encoder import EncodeProfile, duration_of, validate_block

log = get_logger(__name__)

#: Wall-clock seconds the join may take per second of video before it is
#: called hung. The slowest encode measured — libx264 veryfast, 720p, on a
#: Cortex-A53 at 1.4 GHz — ran at about three frames a second, ten seconds
#: of work per second of video; fifteen leaves room for a busy box. A real
#: hang on a ten-minute block is still caught within three hours, and no
#: machine loses its clips to a number chosen on a faster one.
JOIN_SECONDS_PER_VIDEO_SECOND = 15.0

# A transition may not eat more than this share of the shorter neighbouring
# clip, or short clips would be almost entirely crossfade.
MAX_TRANSITION_FRACTION = 0.4

# Where in the block the thumbnail is taken from. Not 0: that is black.
THUMBNAIL_POSITION = 0.12


@dataclass
class BuiltBlock:
    id: str
    path: Path
    duration_s: float
    size_bytes: int
    thumbnail: Path | None
    sidecar: Path
    clips: list[Clip]
    transitions: list[str]
    profile_fingerprint: str
    build_seconds: float
    valid: bool = True
    problems: list[str] = field(default_factory=list)


def plan_transitions(
    clips: list[Clip], cfg: Config, rng: random.Random
) -> tuple[list[float], list[str]]:
    """Durations and named effects for the joins between clips."""
    block = cfg.visual.block
    durations: list[float] = []
    names: list[str] = []
    for index in range(1, len(clips)):
        wanted = rng.uniform(block.transition_min_s, block.transition_max_s)
        limit = MAX_TRANSITION_FRACTION * min(clips[index - 1].duration_s, clips[index].duration_s)
        durations.append(max(0.2, min(wanted, limit)))
        names.append(rng.choice(block.transitions))
    return durations, names


def total_duration(clips: list[Clip], transitions: list[float]) -> float:
    """Clips overlap during a transition, so the sum is not the total."""
    return sum(c.duration_s for c in clips) - sum(transitions)


def build_filtergraph(
    clips: list[Clip],
    transition_s: list[float],
    transition_names: list[str],
    profile: EncodeProfile,
    edge_fade_s: float,
) -> tuple[str, float]:
    """The xfade chain plus the edge fades. Returns (graph, total duration)."""
    steps: list[str] = []

    # Normalise every input first. xfade refuses to work across differing
    # frame rates or timebases, and a stale PTS offset shifts the whole chain.
    for index in range(len(clips)):
        steps.append(
            f"[{index}:v]fps={profile.fps},format={profile.pix_fmt},"
            f"setpts=PTS-STARTPTS,settb=AVTB[c{index}]"
        )

    current = "c0"
    accumulated = clips[0].duration_s
    for index in range(1, len(clips)):
        duration = transition_s[index - 1]
        name = transition_names[index - 1]
        offset = max(0.0, accumulated - duration)
        label = f"x{index}"
        steps.append(
            f"[{current}][c{index}]xfade=transition={name}:"
            f"duration={duration:.3f}:offset={offset:.3f}[{label}]"
        )
        accumulated = accumulated + clips[index].duration_s - duration
        current = label

    total = accumulated
    fade = max(0.0, min(edge_fade_s, total / 4.0))
    if fade > 0:
        steps.append(
            f"[{current}]fade=t=in:st=0:d={fade:.3f},"
            f"fade=t=out:st={max(0.0, total - fade):.3f}:d={fade:.3f}[out]"
        )
    else:
        steps.append(f"[{current}]null[out]")

    return ";".join(steps), total


def build_block(
    clips: list[Clip],
    *,
    cfg: Config,
    profile: EncodeProfile,
    blocks_dir: Path,
    thumbnails_dir: Path,
    rejected_dir: Path,
    source: str = "procedural",
    rng: random.Random | None = None,
    block_id: str | None = None,
) -> BuiltBlock:
    """Concatenate clips into a validated ``.ts`` block."""
    if not clips:
        raise ValueError("a block needs at least one clip")

    started = time.monotonic()
    rng = rng or random.Random()
    block_id = block_id or uuid.uuid4().hex
    blocks_dir.mkdir(parents=True, exist_ok=True)

    transition_s, transition_names = plan_transitions(clips, cfg, rng)
    graph, expected = build_filtergraph(
        clips, transition_s, transition_names, profile, cfg.visual.block.edge_fade_s
    )

    out_path = blocks_dir / f"block_{block_id}.ts"
    argv = [cfg.tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    for clip in clips:
        argv += ["-i", str(clip.path)]
    argv += [
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-an",  # audio comes from Icecast at broadcast time, never from here
        *profile.block_args(),
        "-f",
        "mpegts",
        "-y",
        str(out_path),
    ]

    log.info(
        "building block",
        extra={
            "block": block_id,
            "clips": len(clips),
            "expected_duration_s": round(expected, 1),
            "transitions": transition_names,
        },
    )

    # The join re-encodes the whole block, and how long that takes is a
    # property of the machine, not of the block: two minutes on a desktop,
    # an hour on a small ARM NAS. A fixed timeout threw away an hour and
    # fifty minutes of rendered clips on a TS-230 because the join needed
    # longer than fifteen minutes. So the limit scales with the work.
    timeout_s = max(cfg.tools.timeout_s, expected * JOIN_SECONDS_PER_VIDEO_SECOND)
    try:
        run(
            argv,
            RunOptions(
                timeout_s=timeout_s,
                retries=0,
                nice=cfg.cpu.generator_nice,
                ionice_class=cfg.cpu.generator_ionice_class,
            ),
        )
    except CommandError as exc:
        log.error("block encode failed", extra={"block": block_id, "error": str(exc)})
        out_path.unlink(missing_ok=True)
        raise

    valid, problems = validate_block(
        out_path, profile, expected_duration_s=expected, ffprobe=cfg.tools.ffprobe
    )

    if not valid:
        log.error("block rejected", extra={"block": block_id, "problems": problems})
        rejected_dir.mkdir(parents=True, exist_ok=True)
        target = rejected_dir / out_path.name
        out_path.replace(target)
        out_path = target

    actual = duration_of(out_path, cfg.tools.ffprobe) or expected
    thumbnail = None
    if valid and cfg.visual.pool.thumbnails:
        thumbnail = make_thumbnail(out_path, thumbnails_dir, block_id, actual, cfg)

    block = BuiltBlock(
        id=block_id,
        path=out_path,
        duration_s=actual,
        size_bytes=out_path.stat().st_size if out_path.exists() else 0,
        thumbnail=thumbnail,
        sidecar=out_path.with_suffix(".json"),
        clips=clips,
        transitions=transition_names,
        profile_fingerprint=profile.fingerprint,
        build_seconds=time.monotonic() - started,
        valid=valid,
        problems=problems,
    )
    write_sidecar(block, source=source, profile=profile)

    log.info(
        "block finished",
        extra={
            "block": block_id,
            "valid": valid,
            "duration_s": round(block.duration_s, 1),
            "size_mb": round(block.size_bytes / 1024**2, 1),
            "build_seconds": round(block.build_seconds, 1),
        },
    )
    return block


def make_thumbnail(
    block_path: Path, thumbnails_dir: Path, block_id: str, duration_s: float, cfg: Config
) -> Path | None:
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    target = thumbnails_dir / f"block_{block_id}.jpg"
    position = max(0.0, duration_s * THUMBNAIL_POSITION)
    try:
        run(
            [
                cfg.tools.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-ss",
                f"{position:.3f}",
                "-i",
                str(block_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=480:-2",
                "-q:v",
                "4",
                "-y",
                str(target),
            ],
            RunOptions(timeout_s=120.0),
        )
    except CommandError as exc:
        # A missing thumbnail costs a grey tile in the UI, nothing more.
        log.warning("thumbnail failed", extra={"block": block_id, "error": str(exc)})
        return None
    return target if target.exists() else None


def write_sidecar(block: BuiltBlock, *, source: str, profile: EncodeProfile) -> None:
    payload = {
        "id": block.id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "duration_s": round(block.duration_s, 3),
        "size_bytes": block.size_bytes,
        "valid": block.valid,
        "problems": block.problems,
        "profile": {
            "fingerprint": profile.fingerprint,
            "resolution": profile.resolution,
            "fps": profile.fps,
            "encoder": profile.encoder,
            "bitrate_k": profile.bitrate_k,
            "gop": profile.gop,
            "pix_fmt": profile.pix_fmt,
        },
        "transitions": block.transitions,
        "clips": [clip.as_metadata() for clip in block.clips],
        "build_seconds": round(block.build_seconds, 1),
    }
    block.sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "BuiltBlock",
    "build_block",
    "build_filtergraph",
    "make_thumbnail",
    "plan_transitions",
    "total_duration",
    "write_sidecar",
]
