"""ffmpeg encode profiles and block validation.

The single most important invariant in this project lives here: **every block
in the pool must be encoded with byte-identical parameters**. The streamer
concatenates them with ``-c:v copy``, which does no re-encoding and therefore
cannot reconcile a change in resolution, frame rate, pixel format, profile or
GOP structure. A block that differs is not a slightly worse block; it is a
corrupt stream at the join.

So the arguments are built in exactly one place, and every finished block is
re-read with ffprobe and compared against the profile before it is allowed
into the pool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core import hardware
from app.core.config import Config
from app.core.logging import get_logger
from app.core.proc import CommandError, RunOptions, run

log = get_logger(__name__)

# ffprobe reports H.264 levels as the number times ten: "4.1" is 41.
_LEVEL_SCALE = 10

# Tolerance when comparing a produced duration against the intended one.
DURATION_TOLERANCE_S = 1.5

# Frame rates are compared as floats; 30/1 and 30000/1000 must both pass.
FPS_TOLERANCE = 0.01

# Keyed by (profile fingerprint, encoder setting, ffmpeg path).
_PROFILE_CACHE: dict[tuple[str, str, str], EncodeProfile] = {}


@dataclass(frozen=True)
class EncodeProfile:
    """The one true set of encode parameters, derived from configuration."""

    width: int
    height: int
    fps: int
    bitrate_k: int
    maxrate_k: int
    bufsize_k: int
    gop: int
    keyint_min: int
    bframes: int
    pix_fmt: str
    h264_profile: str
    h264_level: str
    encoder: str
    preset: str
    nvenc_preset: str
    extra_args: tuple[str, ...]
    fingerprint: str

    @classmethod
    def from_config(cls, cfg: Config) -> EncodeProfile:
        """Build the profile, reusing the last one for identical settings.

        Every request that touches the pool needs a profile, and resolving the
        encoder probes ffmpeg. Without this cache the dashboard produced one
        identical "video encoder resolved" line every few seconds — dozens of
        entries that bury the ones that matter.
        """
        key = (cfg.video.profile_fingerprint(), cfg.video.encoder, cfg.tools.ffmpeg)
        cached = _PROFILE_CACHE.get(key)
        if cached is not None:
            return cached

        encoder, reason = hardware.resolve_encoder(cfg.video.encoder, ffmpeg=cfg.tools.ffmpeg)
        log.info("video encoder resolved", extra={"encoder": encoder, "reason": reason})
        profile = cls._build(cfg, encoder)
        _PROFILE_CACHE[key] = profile
        return profile

    @classmethod
    def _build(cls, cfg: Config, encoder: str) -> EncodeProfile:
        return cls(
            width=cfg.video.width,
            height=cfg.video.height,
            fps=cfg.video.fps,
            bitrate_k=cfg.video.bitrate_k,
            maxrate_k=cfg.video.maxrate_k,
            bufsize_k=cfg.video.bufsize_k,
            gop=cfg.video.gop,
            keyint_min=cfg.video.keyint_min,
            bframes=cfg.video.bframes,
            pix_fmt=cfg.video.pix_fmt,
            h264_profile=cfg.video.h264_profile,
            h264_level=cfg.video.h264_level,
            encoder=encoder,
            preset=cfg.video.preset,
            nvenc_preset=cfg.video.nvenc_preset,
            extra_args=tuple(cfg.video.extra_args),
            fingerprint=cfg.video.profile_fingerprint(),
        )

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def level_int(self) -> int:
        return round(float(self.h264_level) * _LEVEL_SCALE)

    def block_args(self) -> list[str]:
        """Encoder arguments for a pool block. Never vary these per block."""
        args = [
            "-c:v",
            self.encoder,
            "-pix_fmt",
            self.pix_fmt,
            "-r",
            str(self.fps),
            "-s",
            self.resolution,
            "-profile:v",
            self.h264_profile,
            "-level",
            self.h264_level,
            "-b:v",
            f"{self.bitrate_k}k",
            "-maxrate",
            f"{self.maxrate_k}k",
            "-bufsize",
            f"{self.bufsize_k}k",
            "-g",
            str(self.gop),
            "-keyint_min",
            str(self.keyint_min),
            "-bf",
            str(self.bframes),
        ]
        if self.encoder == "h264_nvenc":
            # NVENC spells these differently; -sc_threshold does not exist and
            # closed GOPs are requested with -no-scenecut.
            args += ["-preset", self.nvenc_preset, "-no-scenecut", "1", "-rc", "cbr"]
        else:
            # Scene-cut detection would insert keyframes at content-dependent
            # positions, so two blocks would no longer share a GOP structure.
            args += ["-preset", self.preset, "-sc_threshold", "0"]
        args += list(self.extra_args)
        return args

    def intermediate_args(self) -> list[str]:
        """Cheap encoder for throwaway clips.

        Clips are decoded again by the block builder, so quality here only has
        to survive one generation. ultrafast plus a low CRF costs a fraction
        of the final encode and is visually lossless at this bitrate.
        """
        return [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            self.pix_fmt,
            "-g",
            str(self.fps),
        ]


# --------------------------------------------------------------------------
# probing and validation
# --------------------------------------------------------------------------


def probe(path: Path, ffprobe: str = "ffprobe", timeout_s: float = 60.0) -> dict:
    result = run(
        [
            ffprobe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            str(path),
        ],
        RunOptions(timeout_s=timeout_s),
    )
    return json.loads(result.stdout)


def _frame_rate(stream: dict) -> float:
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = stream.get(key) or ""
        if "/" in raw:
            num, _, den = raw.partition("/")
            try:
                numerator, denominator = float(num), float(den)
            except ValueError:
                continue
            if denominator:
                return numerator / denominator
    return 0.0


def validate_block(  # noqa: PLR0912 - one branch per parameter that must match
    path: Path,
    profile: EncodeProfile,
    *,
    expected_duration_s: float | None = None,
    ffprobe: str = "ffprobe",
) -> tuple[bool, list[str]]:
    """Confirm a finished block matches the profile exactly.

    Returns ``(ok, problems)``. Anything in ``problems`` means the block would
    break a ``-c:v copy`` stream at its join and must not enter the pool.
    """
    problems: list[str] = []

    if not path.is_file():
        return False, [f"{path} does not exist"]
    if path.stat().st_size == 0:
        return False, [f"{path} is empty"]

    try:
        info = probe(path, ffprobe)
    except (CommandError, ValueError) as exc:
        return False, [f"ffprobe could not read the file: {exc}"]

    streams = info.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    if len(video) != 1:
        problems.append(f"expected exactly one video stream, found {len(video)}")
    if audio:
        # The streamer supplies audio from Icecast; an audio track here would
        # collide with "-map 0:v -map 1:a".
        problems.append(f"blocks must have no audio track, found {len(audio)}")
    if not video:
        return False, problems

    stream = video[0]

    if stream.get("codec_name") != "h264":
        problems.append(f"codec is {stream.get('codec_name')!r}, expected 'h264'")
    if stream.get("width") != profile.width or stream.get("height") != profile.height:
        problems.append(
            f"resolution is {stream.get('width')}x{stream.get('height')}, "
            f"expected {profile.resolution}"
        )
    if stream.get("pix_fmt") != profile.pix_fmt:
        problems.append(f"pixel format is {stream.get('pix_fmt')!r}, expected {profile.pix_fmt!r}")

    actual_fps = _frame_rate(stream)
    if abs(actual_fps - profile.fps) > FPS_TOLERANCE:
        problems.append(f"frame rate is {actual_fps:.3f}, expected {profile.fps}")

    actual_profile = str(stream.get("profile", ""))
    if actual_profile.lower() != profile.h264_profile.lower():
        problems.append(f"H.264 profile is {actual_profile!r}, expected {profile.h264_profile!r}")

    actual_level = stream.get("level")
    if isinstance(actual_level, int) and actual_level > profile.level_int:
        # A lower level than requested is harmless; a higher one is not, since
        # it means the encoder overrode us and decoders may refuse it.
        problems.append(f"H.264 level is {actual_level / _LEVEL_SCALE:.1f}, above the profile")

    if expected_duration_s is not None:
        try:
            actual_duration = float(info.get("format", {}).get("duration", 0.0))
        except (TypeError, ValueError):
            actual_duration = 0.0
        if abs(actual_duration - expected_duration_s) > DURATION_TOLERANCE_S:
            problems.append(
                f"duration is {actual_duration:.1f}s, expected about {expected_duration_s:.1f}s"
            )

    return not problems, problems


def duration_of(path: Path, ffprobe: str = "ffprobe") -> float:
    try:
        info = probe(path, ffprobe)
        return float(info.get("format", {}).get("duration", 0.0))
    except (CommandError, ValueError, TypeError):
        return 0.0


__all__ = [
    "DURATION_TOLERANCE_S",
    "EncodeProfile",
    "duration_of",
    "probe",
    "validate_block",
]
