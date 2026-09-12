"""Turn a generator into an encoded video clip.

Frames are produced in numpy at a reduced resolution and streamed as raw
RGB24 straight into ffmpeg's stdin — no PNG files, no temporary frame
directory. ffmpeg upscales with lanczos and applies the finishing effects,
because doing grain and vignette in numpy would cost more than the encode.

**Perfect loops.** Four of the six generators are periodic by construction and
loop exactly with no extra work. The two stateful ones (particle trails,
reaction-diffusion) cannot: their output is history. For those, the clip is
rendered twice.

Rendering twice sounds wasteful until you look at the alternative. A seamless
loop needs the tail blended onto the head, and the head has to be emitted
first — so a single pass through a pipe would have to buffer the entire clip
(around 600 MB at 640x360, far more at full resolution). Instead the first
pass keeps only the last B frames, and the second pass — from a fresh,
identically seeded generator, so it reproduces the same sequence exactly —
blends them into the head as it streams. Bounded memory, roughly 20 MB.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.config import Config
from app.core.logging import get_logger
from app.core.proc import PipedCommand
from app.services.visual.encoder import EncodeProfile, duration_of
from app.services.visual.palettes import Palette
from app.services.visual.procedural.base import ProceduralGenerator, RenderContext

log = get_logger(__name__)

# Longest tail blended into the head when forcing a loop, in seconds.
MAX_LOOP_BLEND_S = 1.0
# ...and never more than this fraction of the clip, so a short clip is not
# mostly crossfade.
MAX_LOOP_BLEND_FRACTION = 0.25

# ffmpeg's noise filter takes 0-100.
MAX_GRAIN = 100


@dataclass
class Clip:
    """One finished intermediate clip on disk."""

    path: Path
    duration_s: float
    generator: str
    palette: str
    seed: int
    loops: bool
    frames: int
    render_size: tuple[int, int]
    build_seconds: float = 0.0
    params: dict = field(default_factory=dict)

    def as_metadata(self) -> dict:
        return {
            "file": self.path.name,
            "generator": self.generator,
            "palette": self.palette,
            "seed": self.seed,
            "duration_s": round(self.duration_s, 3),
            "frames": self.frames,
            "loops": self.loops,
            "render_size": f"{self.render_size[0]}x{self.render_size[1]}",
            "build_seconds": round(self.build_seconds, 1),
            "params": self.params,
        }


def render_size(cfg: Config) -> tuple[int, int]:
    """Generator resolution: the output scaled down, rounded to even pixels."""
    scale = cfg.visual.procedural.render_scale
    width = max(64, int(cfg.video.width * scale) // 2 * 2)
    height = max(64, int(cfg.video.height * scale) // 2 * 2)
    return width, height


def loop_blend_frames(total_frames: int, fps: int) -> int:
    return max(1, min(int(fps * MAX_LOOP_BLEND_S), int(total_frames * MAX_LOOP_BLEND_FRACTION)))


def build_filter_chain(cfg: Config, profile: EncodeProfile) -> str:
    """Scale to output resolution, then the finishing effects.

    Order matters: effects run after the upscale so that grain is grain at
    output resolution rather than smeared blocks, and the chromatic offset is
    measured in output pixels.
    """
    clip_cfg = cfg.visual.clip
    parts = [f"scale={profile.width}:{profile.height}:flags=lanczos"]

    if clip_cfg.chromatic_aberration > 0:
        shift = max(1, round(clip_cfg.chromatic_aberration * profile.width))
        parts.append(f"rgbashift=rh=-{shift}:bh={shift}")
    if clip_cfg.vignette:
        parts.append("vignette=PI/5")
    if clip_cfg.film_grain > 0:
        strength = max(1, min(MAX_GRAIN, round(clip_cfg.film_grain * 100)))
        parts.append(f"noise=alls={strength}:allf=t+u")

    parts.append(f"format={profile.pix_fmt}")
    return ",".join(parts)


def _ffmpeg_argv(
    cfg: Config,
    profile: EncodeProfile,
    width: int,
    height: int,
    out_path: Path,
) -> list[str]:
    return [
        cfg.tools.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(profile.fps),
        "-i",
        "pipe:0",
        "-vf",
        build_filter_chain(cfg, profile),
        *profile.intermediate_args(),
        "-an",
        "-y",
        str(out_path),
    ]


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend in float, returned as uint8 without wraparound."""
    mixed = base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


def render_clip(
    factory: Callable[[], ProceduralGenerator],
    *,
    cfg: Config,
    profile: EncodeProfile,
    palette: Palette,
    duration_s: float,
    seed: int,
    out_dir: Path,
    force_loop: bool | None = None,
    progress: callable | None = None,
) -> Clip:
    """Render one clip and return its description.

    ``factory`` is anything that returns a fresh generator: a generator class
    works, and so does a lambda holding already-loaded AI stills. It has to be
    callable more than once, because forcing a loop renders the clip twice.

    ``progress`` is called as ``progress(frame, total)`` if given.
    """
    started = time.monotonic()
    width, height = render_size(cfg)
    fps = profile.fps
    total_frames = max(1, round(duration_s * fps))

    # One throwaway instance, to read the generator's own properties. Cheap:
    # nothing is allocated until prepare() runs.
    prototype = factory()
    generator_name = prototype.name
    loops_natively = prototype.loops_natively

    if force_loop is None:
        force_loop = cfg.visual.clip.perfect_loop
    needs_blend = force_loop and not loops_natively

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clip_{uuid.uuid4().hex}.mkv"

    def make_context() -> tuple[ProceduralGenerator, RenderContext]:
        import random

        generator = factory()
        ctx = RenderContext(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            palette=palette,
            seed=seed,
            rng=random.Random(seed),
        )
        generator.prepare(ctx)
        return generator, ctx

    tail: list[np.ndarray] = []
    blend = 0
    if needs_blend:
        blend = loop_blend_frames(total_frames, fps)
        log.debug(
            "pre-rendering the loop tail",
            extra={"generator": generator_name, "blend_frames": blend},
        )
        generator, ctx = make_context()
        for frame in range(total_frames + blend):
            image = generator.render(frame, ctx)
            if frame >= total_frames:
                tail.append(image.copy())

    generator, ctx = make_context()
    argv = _ffmpeg_argv(cfg, profile, width, height, out_path)

    with PipedCommand(
        argv,
        timeout_s=cfg.tools.timeout_s,
        nice=cfg.cpu.generator_nice,
        ionice_class=cfg.cpu.generator_ionice_class,
    ) as pipe:
        for frame in range(total_frames):
            image = generator.render(frame, ctx)
            if tail and frame < blend:
                # Alpha runs 1 -> 0 across the blend, so the clip opens on the
                # tail and has arrived at its own head by the end of it.
                alpha = 1.0 - (frame / blend)
                image = _blend(image, tail[frame], alpha)
            if image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{generator_name} returned {image.shape} {image.dtype}, "
                    f"expected ({height}, {width}, 3) uint8"
                )
            pipe.write(np.ascontiguousarray(image).tobytes())
            if progress is not None and frame % fps == 0:
                progress(frame, total_frames)

    actual = duration_of(out_path, cfg.tools.ffprobe)
    clip = Clip(
        path=out_path,
        duration_s=actual or (total_frames / fps),
        generator=generator_name,
        palette=palette.id,
        seed=seed,
        loops=loops_natively or needs_blend,
        frames=total_frames,
        render_size=(width, height),
        build_seconds=time.monotonic() - started,
        params=dict(generator.describe(ctx)),
    )
    log.info(
        "clip rendered",
        extra={
            "generator": clip.generator,
            "palette": clip.palette,
            "seconds": round(clip.build_seconds, 1),
            "duration_s": round(clip.duration_s, 1),
            "fps_rendered": round(total_frames / max(clip.build_seconds, 1e-6), 1),
        },
    )
    return clip


def pick_clip_duration(cfg: Config, rng) -> float:
    lo = cfg.visual.clip.min_duration_s
    hi = cfg.visual.clip.max_duration_s
    return rng.uniform(lo, hi) if hi > lo else lo


def frames_for(duration_s: float, fps: int) -> int:
    return max(1, math.ceil(duration_s * fps))


__all__ = [
    "Clip",
    "build_filter_chain",
    "frames_for",
    "loop_blend_frames",
    "pick_clip_duration",
    "render_clip",
    "render_size",
]
