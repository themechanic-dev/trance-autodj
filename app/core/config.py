"""Configuration: YAML file, overridden by TAD_* environment variables.

The whole application reads its settings from here. Nothing else parses YAML
and nothing else looks at os.environ for tunables, so there is exactly one
place to look when a value is not what you expected.

Precedence, lowest to highest:
    1. the defaults declared on the models below
    2. config/config.yaml
    3. TAD_* environment variables

Environment names mirror the nesting with a double underscore, e.g.
``TAD_AUDIO__CROSSFADE__DURATION_S=18``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ENV_PREFIX = "TAD_"
ENV_NESTED_DELIMITER = "__"

# TAD_* variables that address the process rather than a setting. They must be
# stripped before validation, because unknown keys are a hard error and these
# are ones we document and set ourselves.
RESERVED_ENV_KEYS = frozenset({"config_file", "master_key", "service"})

MAX_HOUR = 23
MAX_MINUTE = 59

LogLevel = Literal["debug", "info", "warning", "error"]
LogFormat = Literal["json", "console"]
PlayMode = Literal["shuffle", "sequential", "random"]
FadeCurve = Literal["linear", "log", "exponential", "sinusoidal"]
AudioFormat = Literal["mp3", "opus"]
VideoEncoder = Literal["auto", "libx264", "h264_nvenc"]
VideoMode = Literal["copy", "reencode"]
# What the station plays: the one loaded list on a loop, or several lists
# one after another in an order the operator arranges.
PlaylistPlayback = Literal["single", "rotation"]
AiBackend = Literal["auto", "cuda", "openvino", "sdcpp", "none"]
IoClass = Literal["none", "realtime", "best-effort", "idle"]


class _Model(BaseModel):
    """Base: reject unknown keys so a typo in config.yaml is a loud error."""

    model_config = {"extra": "forbid", "validate_assignment": True}


# --------------------------------------------------------------------------
# app / auth
# --------------------------------------------------------------------------


class AppConfig(_Model):
    # Binds every interface on purpose: in a container there is nothing
    # else to bind, and reaching it from outside is the whole point.
    host: str = "0.0.0.0"  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65535)
    data_dir: Path = Path("./data")
    config_dir: Path = Path("./config")
    timezone: str = "Europe/Athens"
    log_level: LogLevel = "info"
    log_format: LogFormat = "json"
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=4096)
    log_backup_count: int = Field(default=7, ge=0)
    # Come back up the way we went down. A station that needs someone to open
    # a browser after every restart is not a 24/7 station, and the container
    # is configured to restart itself.
    resume_on_start: bool = True


class AuthConfig(_Model):
    enabled: bool = True
    username: str = "admin"
    password_hash: str = ""
    session_secret: str = ""
    session_max_age_s: int = Field(default=7 * 24 * 3600, ge=60)
    cookie_secure: bool = False


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------


class CrossfadeConfig(_Model):
    duration_s: float = Field(default=14.0, gt=0)
    min_duration_s: float = Field(default=4.0, gt=0)
    max_duration_s: float = Field(default=30.0, gt=0)
    minimum_track_s: float = Field(default=2.0, ge=0)
    fade_in_curve: FadeCurve = "log"
    fade_out_curve: FadeCurve = "log"
    # Width of Liquidsoap's power-measurement window. This is what makes the
    # transition loudness-aware rather than blindly symmetric.
    width_s: float = Field(default=2.0, gt=0)
    # Round each crossfade to a whole number of bars of the outgoing track,
    # using its measured tempo. Off by default: it only does anything once
    # tracks have been analysed, and it changes how the station sounds.
    beat_aligned: bool = False
    # Also trim the start of each track to its first beat, so the two grids
    # line up rather than merely running at the same speed. This *removes*
    # audio, which is why it is separate and off by default.
    beat_align_cue_in: bool = False
    # Never trim more than this looking for the first beat.
    max_cue_in_s: float = Field(default=2.0, ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> CrossfadeConfig:
        if self.min_duration_s > self.max_duration_s:
            raise ValueError("crossfade.min_duration_s must be <= max_duration_s")
        if not self.min_duration_s <= self.duration_s <= self.max_duration_s:
            raise ValueError(
                f"crossfade.duration_s ({self.duration_s}) must lie within "
                f"[{self.min_duration_s}, {self.max_duration_s}]"
            )
        return self


class LoudnessConfig(_Model):
    enabled: bool = True
    target_lufs: float = Field(default=-14.0, le=0)


class IcecastConfig(_Model):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    mount: str = "/stream"
    format: AudioFormat = "mp3"
    bitrate_k: int = Field(default=320, ge=32, le=512)
    source_password: str = ""
    admin_password: str = ""
    managed: bool = True
    binary: str = "icecast2"

    @field_validator("mount")
    @classmethod
    def _leading_slash(cls, v: str) -> str:
        return v if v.startswith("/") else "/" + v

    @property
    def stream_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.mount}"


class LiquidsoapConfig(_Model):
    binary: str = "liquidsoap"
    telnet_host: str = "127.0.0.1"
    telnet_port: int = Field(default=1234, ge=1, le=65535)
    extra_config: str = ""


class AnalysisConfig(_Model):
    """Offline tempo and beat detection, run when a file is scanned."""

    enabled: bool = True
    # Anything outside this is not trance, and accepting it would mostly mean
    # accepting a half- or double-tempo mistake.
    min_bpm: float = Field(default=118.0, gt=0)
    max_bpm: float = Field(default=152.0, gt=0)
    # Use librosa when it is installed. It is more general than the built-in
    # detector but pulls in scipy, scikit-learn and numba.
    prefer_librosa: bool = True
    # Below this confidence the measurement is ignored rather than trusted.
    min_confidence: float = Field(default=0.12, ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> AnalysisConfig:
        if self.min_bpm >= self.max_bpm:
            raise ValueError("analysis.min_bpm must be below analysis.max_bpm")
        return self


class AudioConfig(_Model):
    mode: PlayMode = "shuffle"
    # "single" plays the one activated list and loops it. "rotation" plays the
    # lists the operator put in the rotation, in that order, then starts over.
    #
    # The order is ours to keep, not Liquidsoap's: in rotation the generated
    # script reads the file straight through, and this application decides
    # what order the file is in. That is the only way a per-list order can
    # survive a global shuffle setting.
    playlist_playback: PlaylistPlayback = "single"
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    crossfade: CrossfadeConfig = Field(default_factory=CrossfadeConfig)
    loudness: LoudnessConfig = Field(default_factory=LoudnessConfig)
    safe_track: str = ""
    retry_delay_s: float = Field(default=10.0, gt=0)
    icecast: IcecastConfig = Field(default_factory=IcecastConfig)
    liquidsoap: LiquidsoapConfig = Field(default_factory=LiquidsoapConfig)


# --------------------------------------------------------------------------
# video profile
# --------------------------------------------------------------------------


class VideoConfig(_Model):
    width: int = Field(default=1280, ge=64)
    height: int = Field(default=720, ge=64)
    fps: int = Field(default=30, ge=1, le=120)
    bitrate_k: int = Field(default=3500, ge=100)
    maxrate_k: int = Field(default=3500, ge=100)
    bufsize_k: int = Field(default=7000, ge=100)
    gop: int = Field(default=60, ge=1)
    keyint_min: int = Field(default=60, ge=1)
    bframes: int = Field(default=2, ge=0, le=16)
    pix_fmt: str = "yuv420p"
    h264_profile: str = "high"
    h264_level: str = "4.1"
    encoder: VideoEncoder = "auto"
    preset: str = "veryfast"
    nvenc_preset: str = "p4"
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_even_dimensions(self) -> VideoConfig:
        # yuv420p subsamples chroma by 2; odd dimensions are rejected by x264.
        if self.width % 2 or self.height % 2:
            raise ValueError("video.width and video.height must both be even")
        return self

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def profile_fingerprint(self) -> str:
        """Identity of the encode profile.

        Every block in the pool must share this. It is stored in each block's
        sidecar so the pool can reject anything produced under older settings
        instead of letting the streamer discover it live.
        """
        parts = {
            "w": self.width,
            "h": self.height,
            "fps": self.fps,
            "gop": self.gop,
            "keyint_min": self.keyint_min,
            "bf": self.bframes,
            "pix_fmt": self.pix_fmt,
            "profile": self.h264_profile,
            "level": self.h264_level,
            "b": self.bitrate_k,
        }
        return json.dumps(parts, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------
# visual generator
# --------------------------------------------------------------------------


class ClipConfig(_Model):
    min_duration_s: float = Field(default=20.0, gt=0)
    max_duration_s: float = Field(default=40.0, gt=0)
    perfect_loop: bool = True
    film_grain: float = Field(default=0.04, ge=0, le=1)
    vignette: bool = True
    chromatic_aberration: float = Field(default=0.0015, ge=0, le=0.1)

    @model_validator(mode="after")
    def _check_range(self) -> ClipConfig:
        if self.min_duration_s > self.max_duration_s:
            raise ValueError("clip.min_duration_s must be <= clip.max_duration_s")
        return self


class BlockConfig(_Model):
    duration_s: float = Field(default=600.0, gt=0)
    transition_min_s: float = Field(default=3.0, gt=0)
    transition_max_s: float = Field(default=5.0, gt=0)
    transitions: list[str] = Field(
        default_factory=lambda: [
            "fade",
            "dissolve",
            "smoothleft",
            "circleopen",
            "radial",
            "pixelize",
        ]
    )
    edge_fade_s: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> BlockConfig:
        if self.transition_min_s > self.transition_max_s:
            raise ValueError("block.transition_min_s must be <= transition_max_s")
        if not self.transitions:
            raise ValueError("block.transitions must not be empty")
        return self


class PoolConfig(_Model):
    min_blocks: int = Field(default=12, ge=1)
    max_blocks: int = Field(default=60, ge=1)
    max_disk_gb: float = Field(default=50.0, gt=0)
    keep_rejected: bool = True
    thumbnails: bool = True

    @model_validator(mode="after")
    def _check(self) -> PoolConfig:
        if self.min_blocks > self.max_blocks:
            raise ValueError("pool.min_blocks must be <= pool.max_blocks")
        return self


class SourcesConfig(_Model):
    ai_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class AiConfig(_Model):
    backend: AiBackend = "auto"
    model_id: str = "stabilityai/sd-turbo"
    models_dir: str = ""
    steps: int = Field(default=4, ge=1, le=150)
    guidance_scale: float = Field(default=0.0, ge=0.0)
    width: int = Field(default=512, ge=64)
    height: int = Field(default=512, ge=64)
    threads: int = Field(default=0, ge=0)
    negative_prompt: str = "text, watermark, logo, face, human, blurry, low quality, jpeg artifacts"
    images_per_clip: int = Field(default=3, ge=1, le=16)


class ProceduralConfig(_Model):
    generators: list[str] = Field(
        default_factory=lambda: [
            "flowfield",
            "plasma",
            "domainwarp",
            "tunnel",
            "reaction_diffusion",
            "waves",
        ]
    )
    render_scale: float = Field(default=0.5, gt=0, le=1.0)
    use_numba: bool = True


class VisualConfig(_Model):
    clip: ClipConfig = Field(default_factory=ClipConfig)
    block: BlockConfig = Field(default_factory=BlockConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    ai: AiConfig = Field(default_factory=AiConfig)
    procedural: ProceduralConfig = Field(default_factory=ProceduralConfig)


# --------------------------------------------------------------------------
# streamer
# --------------------------------------------------------------------------


class YoutubeConfig(_Model):
    rtmp_url: str = "rtmp://a.rtmp.youtube.com/live2"
    stream_key: str = ""
    # The name each broadcast gets, applied when it goes live. RTMP carries no
    # title — it is only video and audio — so this is done through the Data
    # API afterwards, and only if the account has been connected.
    #
    # {date}, {time} and {datetime} are replaced at the moment of going live,
    # in the configured timezone. An empty template leaves whatever YouTube
    # already had, which is the right default for anyone not connecting an
    # account.
    title_template: str = ""
    description_template: str = ""


class StreamAudioConfig(_Model):
    codec: str = "aac"
    bitrate_k: int = Field(default=192, ge=32, le=512)
    sample_rate: int = Field(default=44100, ge=8000)
    channels: int = Field(default=2, ge=1, le=2)


class WatchdogConfig(_Model):
    enabled: bool = True
    backoff_min_s: float = Field(default=2.0, gt=0)
    backoff_max_s: float = Field(default=60.0, gt=0)
    stall_timeout_s: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> WatchdogConfig:
        if self.backoff_min_s > self.backoff_max_s:
            raise ValueError("watchdog.backoff_min_s must be <= backoff_max_s")
        return self


class ReactiveOverlayConfig(_Model):
    enabled: bool = False
    filter: Literal["showcqt", "showspectrum", "showwaves"] = "showcqt"
    opacity: float = Field(default=0.25, ge=0, le=1)
    height_pct: int = Field(default=22, ge=1, le=100)


class StreamConfig(_Model):
    youtube: YoutubeConfig = Field(default_factory=YoutubeConfig)
    audio: StreamAudioConfig = Field(default_factory=StreamAudioConfig)
    video_mode: VideoMode = "copy"
    fifo_name: str = "video.fifo"
    play_count_weight: float = Field(default=1.0, ge=0)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
    reactive_overlay: ReactiveOverlayConfig = Field(default_factory=ReactiveOverlayConfig)

    @model_validator(mode="after")
    def _overlay_forces_reencode(self) -> StreamConfig:
        if self.reactive_overlay.enabled and self.video_mode == "copy":
            raise ValueError(
                "stream.reactive_overlay.enabled requires stream.video_mode: reencode "
                "— compositing an overlay cannot be done on a copied video stream"
            )
        return self


# --------------------------------------------------------------------------
# cpu policy
# --------------------------------------------------------------------------


class QuietHoursConfig(_Model):
    enabled: bool = False
    start: str = "02:00"
    end: str = "08:00"

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        try:
            hh, mm = v.split(":")
            if not (0 <= int(hh) <= MAX_HOUR and 0 <= int(mm) <= MAX_MINUTE):
                raise ValueError
        except Exception as exc:
            raise ValueError(f"quiet_hours time must be HH:MM, got {v!r}") from exc
        return v


class CpuConfig(_Model):
    generator_nice: int = Field(default=19, ge=-20, le=19)
    generator_ionice_class: IoClass = "idle"
    generator_quota_pct: int = Field(default=150, ge=1)
    load_threshold_factor: float = Field(default=0.7, gt=0)
    throttle_poll_s: float = Field(default=30.0, gt=0)
    max_concurrent_jobs: int = Field(default=1, ge=1)
    pause_during_stream: bool = False
    quiet_hours: QuietHoursConfig = Field(default_factory=QuietHoursConfig)


class ToolsConfig(_Model):
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    timeout_s: float = Field(default=900.0, gt=0)
    retries: int = Field(default=2, ge=0)
    retry_delay_s: float = Field(default=3.0, ge=0)


# --------------------------------------------------------------------------
# root
# --------------------------------------------------------------------------


class Config(_Model):
    app: AppConfig = Field(default_factory=AppConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    cpu: CpuConfig = Field(default_factory=CpuConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    # Filled in by load_config(); not part of the file itself.
    source_path: Path | None = Field(default=None, exclude=True)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(raw: str) -> Any:
    """Turn an environment string into the value it obviously represents.

    JSON first (so lists and objects work), then the usual boolean words, then
    the string itself. Pydantic does the real type checking afterwards.
    """
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a nested dict from every TAD_* variable."""
    environ = os.environ if environ is None else environ
    result: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = name[len(ENV_PREFIX) :].lower().split(ENV_NESTED_DELIMITER)
        if not path or not path[0]:
            continue
        cursor = result
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce_scalar(raw)
    return result


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Config:
    """Read config.yaml (if present) and apply TAD_* overrides.

    A missing file is not an error: the defaults plus environment are a valid
    configuration, which is what makes a bare ``docker run`` work.
    """
    if path is None:
        path = os.environ.get(f"{ENV_PREFIX}CONFIG_FILE", "config/config.yaml")
    config_path = Path(path)

    data: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping at the top level")
        data = loaded

    overrides = env_overrides(environ)
    for reserved in RESERVED_ENV_KEYS:
        overrides.pop(reserved, None)
    merged = _deep_merge(data, overrides)

    cfg = Config.model_validate(merged)
    cfg.source_path = config_path if config_path.is_file() else None
    return cfg


__all__ = [
    "ENV_PREFIX",
    "RESERVED_ENV_KEYS",
    "Config",
    "env_overrides",
    "load_config",
]
