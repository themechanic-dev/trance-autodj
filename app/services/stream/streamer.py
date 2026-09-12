"""The live broadcast: block bytes plus Icecast audio, out to RTMP.

    feeder -> FIFO ─┐
                    ├─> ffmpeg ─> rtmp://…/live2/<key> -> YouTube
    icecast ────────┘

ffmpeg does no video work at all. ``-c:v copy`` moves the already-encoded
bytes straight through, so the only real cost is one AAC audio encode. That is
the point of the whole architecture, and the reason a 24/7 stream fits on a
CPU-only VM.

Health comes from ``-progress pipe:1``, which emits ``key=value`` lines. That
is parsed rather than the human-readable stderr, because the stderr format is
not a contract and changes between releases.
"""

from __future__ import annotations

import contextlib
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

from app.core.config import Config
from app.core.logging import get_logger, redact, register_secret
from app.core.paths import Paths
from app.services.visual.encoder import EncodeProfile

log = get_logger(__name__)

# stderr lines kept for the dashboard.
LOG_TAIL = 200

# Consider the process healthy (reset the backoff) after this long.
HEALTHY_AFTER_S = 60.0

# Progress fields we read. Anything else ffmpeg emits is ignored.
_PROGRESS_KEYS = frozenset(
    {"frame", "fps", "bitrate", "drop_frames", "dup_frames", "speed", "out_time_us"}
)


@dataclass
class StreamHealth:
    frames: int = 0
    fps: float = 0.0
    bitrate_kbits: float = 0.0
    dropped_frames: int = 0
    duplicated_frames: int = 0
    speed: float = 0.0
    out_time_s: float = 0.0
    updated_at: float = 0.0

    @property
    def stale_s(self) -> float:
        return time.time() - self.updated_at if self.updated_at else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "frames": self.frames,
            "fps": round(self.fps, 1),
            "bitrate_kbits": round(self.bitrate_kbits, 1),
            "dropped_frames": self.dropped_frames,
            "duplicated_frames": self.duplicated_frames,
            "speed": round(self.speed, 3),
            "out_time_s": round(self.out_time_s, 1),
            "stale_s": round(self.stale_s, 1),
        }


def _pipeline(cfg: Config, paths: Paths, profile: EncodeProfile, *, progress: bool) -> list[str]:
    """Everything except where the result goes.

    The broadcast and the monitor share this on purpose: same inputs, same
    codecs, same bitrates. A preview that is encoded differently from the
    thing it previews is worse than no preview, because it invites you to
    trust it.
    """
    stream_cfg = cfg.stream
    audio_url = cfg.audio.icecast.stream_url

    argv = [
        cfg.tools.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        # Structured progress on stdout; the human-readable stats are noise.
        # The monitor cannot have it: stdout is carrying the video there.
        "-nostats",
        *(["-progress", "pipe:1"] if progress else []),
        # -re paces the input at wall-clock speed. Without it ffmpeg would
        # read the pipe as fast as the feeder can fill it and race ahead of
        # real time until YouTube rejected the stream.
        "-re",
        "-f",
        "mpegts",
        # Blocks are independently encoded files concatenated as raw bytes.
        # Their timestamps restart at each join; +genpts rebases them. Without
        # this the second block onwards is simply not played.
        "-fflags",
        "+genpts",
        "-i",
        str(paths.video_fifo),
        # Audio from Icecast. Reconnect flags matter: Liquidsoap restarting
        # must not end the broadcast.
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "10",
        # Those three only cover a stream that drops after it opened. Icecast
        # answers 404 for a mount that has no source yet, and a 404 at open
        # time killed ffmpeg outright — which is exactly what happens in the
        # seconds after a restart, when the dashboard already reports the
        # audio as on air but Liquidsoap has not finished connecting. Retry
        # the ones worth retrying instead of giving up.
        "-reconnect_on_http_error",
        "404,403,429,500,502,503,504",
        "-reconnect_on_network_error",
        "1",
        "-i",
        audio_url,
        "-map",
        "0:v",
        "-map",
        "1:a",
    ]

    if stream_cfg.video_mode == "copy" and not stream_cfg.reactive_overlay.enabled:
        argv += ["-c:v", "copy"]
    else:
        # The escape hatch, and the only mode in which an overlay is possible.
        argv += ["-c:v", profile.encoder]
        argv += profile.block_args()[2:]  # everything after "-c:v <encoder>"
        if stream_cfg.reactive_overlay.enabled:
            argv += ["-filter_complex", _overlay_filter(cfg, profile)]
            argv += ["-map", "[vout]"]

    argv += [
        "-c:a",
        stream_cfg.audio.codec,
        "-b:a",
        f"{stream_cfg.audio.bitrate_k}k",
        "-ar",
        str(stream_cfg.audio.sample_rate),
        "-ac",
        str(stream_cfg.audio.channels),
    ]
    return argv


def build_command(
    cfg: Config,
    paths: Paths,
    profile: EncodeProfile,
    *,
    stream_key: str,
) -> list[str]:
    """The ffmpeg command line. One place, so it can be shown and tested."""
    target = f"{cfg.stream.youtube.rtmp_url.rstrip('/')}/{stream_key}"
    # YouTube wants a keyframe-aligned FLV stream.
    return [*_pipeline(cfg, paths, profile, progress=True), "-f", "flv", target]


def build_monitor_command(cfg: Config, paths: Paths, profile: EncodeProfile) -> list[str]:
    """The same broadcast, muxed for a browser instead of for YouTube.

    Fragmented MP4 on stdout: the browser can start playing the first fragment
    without an index at the end of the file, which a normal MP4 would need and
    a live stream can never have. Everything before the container is identical
    to what goes on air.
    """
    return [
        *_pipeline(cfg, paths, profile, progress=False),
        "-f",
        "mp4",
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        # Small fragments so playback starts quickly instead of after a
        # keyframe interval's worth of buffering.
        "-frag_duration",
        "500000",
        "pipe:1",
    ]


def _overlay_filter(cfg: Config, profile: EncodeProfile) -> str:
    overlay = cfg.stream.reactive_overlay
    height = max(16, profile.height * overlay.height_pct // 100)
    return (
        f"[1:a]{overlay.filter}=s={profile.width}x{height}[viz];"
        f"[viz]format=yuva420p,colorchannelmixer=aa={overlay.opacity}[vizt];"
        f"[0:v][vizt]overlay=0:H-h:format=auto[vout]"
    )


@dataclass
class StreamerStatus:
    live: bool
    since: float | None
    restarts: int
    last_error: str
    health: dict
    command: str

    def as_dict(self) -> dict[str, object]:
        return {
            "live": self.live,
            "uptime_s": round(time.time() - self.since, 1) if self.since else 0.0,
            "restarts": self.restarts,
            "last_error": self.last_error,
            "health": self.health,
            "command": self.command,
        }


class FfmpegStreamer:
    """Runs the broadcast process and keeps it running."""

    def __init__(self, cfg: Config, paths: Paths, profile: EncodeProfile) -> None:
        self.cfg = cfg
        self.paths = paths
        self.profile = profile
        self.health = StreamHealth()
        self._process: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lines: list[str] = []
        self._started_at: float | None = None
        self._restarts = 0
        self._last_error = ""
        self._command: list[str] = []
        self._on_exit: callable | None = None

    # -- process -----------------------------------------------------------

    def start(self, stream_key: str, *, on_exit=None) -> None:
        register_secret(stream_key)
        self._stop.clear()
        self._on_exit = on_exit
        self._command = build_command(self.cfg, self.paths, self.profile, stream_key=stream_key)
        log.info("starting the broadcast", extra={"command": redact(" ".join(self._command))})

        self._process = subprocess.Popen(  # noqa: S603 - argv is a list
            self._command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._started_at = time.time()
        self.health = StreamHealth()

        self._threads = [
            threading.Thread(target=self._read_progress, name="ffmpeg-progress", daemon=True),
            threading.Thread(target=self._read_stderr, name="ffmpeg-stderr", daemon=True),
            threading.Thread(target=self._wait, name="ffmpeg-wait", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _read_progress(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            self._apply_progress(key.strip(), value.strip())

    def _apply_progress(self, key: str, value: str) -> None:
        """Fold one ``key=value`` line into the health snapshot.

        The clock is stamped only when a value was genuinely understood.
        Stamping it for every line — including the "N/A" fields ffmpeg emits
        before the first frame — would make the stall watchdog blind to a
        process that is running but has stopped producing anything.
        """
        if key not in _PROGRESS_KEYS:
            return

        try:
            if key == "frame":
                self.health.frames = int(value)
            elif key == "fps":
                self.health.fps = float(value)
            elif key == "bitrate":
                if not value.endswith("kbits/s"):
                    return
                self.health.bitrate_kbits = float(value.removesuffix("kbits/s"))
            elif key == "drop_frames":
                self.health.dropped_frames = int(value)
            elif key == "dup_frames":
                self.health.duplicated_frames = int(value)
            elif key == "speed":
                if not value.endswith("x"):
                    return
                self.health.speed = float(value.removesuffix("x"))
            elif key == "out_time_us":
                self.health.out_time_s = int(value) / 1_000_000
            else:  # pragma: no cover - _PROGRESS_KEYS and this list agree
                return
        except ValueError:
            # "N/A" appears in every numeric field before the first frame.
            return

        self.health.updated_at = time.time()

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for raw in process.stderr:
            line = redact(raw.decode("utf-8", errors="replace").rstrip())
            if not line:
                continue
            self._lines.append(line)
            del self._lines[:-LOG_TAIL]
            lowered = line.lower()
            if "error" in lowered or "failed" in lowered or "invalid" in lowered:
                self._last_error = line
                log.warning("ffmpeg: %s", line)
            else:
                log.debug("ffmpeg: %s", line)

    def _wait(self) -> None:
        process = self._process
        if process is None:
            return
        code = process.wait()
        if self._stop.is_set():
            return
        log.error(
            "the broadcast process exited",
            extra={"exit_code": code, "last_error": self._last_error},
        )
        if self._on_exit is not None:
            self._on_exit(code)

    def stop(self, grace_s: float = 20.0) -> None:
        """Stop cleanly so YouTube sees the stream end rather than hang.

        SIGINT, not SIGTERM: ffmpeg treats SIGINT as "finish and flush", which
        closes the RTMP session properly. A killed process leaves the ingest
        waiting for data that will never come, and YouTube keeps the broadcast
        in a stuck state for minutes.
        """
        self._stop.set()
        process = self._process
        if process is None or process.poll() is not None:
            return
        log.info("stopping the broadcast")
        with contextlib.suppress(ProcessLookupError, OSError):
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg did not stop on SIGINT; terminating")
            with contextlib.suppress(ProcessLookupError, OSError):
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.kill()

    # -- introspection -----------------------------------------------------

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def uptime_s(self) -> float:
        return time.time() - self._started_at if (self.running and self._started_at) else 0.0

    def status(self) -> StreamerStatus:
        return StreamerStatus(
            live=self.running,
            since=self._started_at if self.running else None,
            restarts=self._restarts,
            last_error=self._last_error,
            health=self.health.as_dict(),
            command=redact(" ".join(self._command)),
        )

    def tail(self, count: int = 50) -> list[str]:
        return self._lines[-count:]

    def note_restart(self) -> None:
        self._restarts += 1


__all__ = [
    "FfmpegStreamer",
    "StreamHealth",
    "StreamerStatus",
    "build_command",
]
