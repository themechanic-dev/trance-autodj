"""Preflight: is this machine actually able to run the thing?

Run from the CLI (``python -m scripts.preflight``) and from the dashboard on
first start. The point is to fail in one readable report rather than three
days later at 04:00 with a cryptic ffmpeg error, so every check carries a hint
that says what to do about it.

Statuses:
    ok   — verified working
    warn — degraded but the system runs (e.g. no AI backend: procedural only)
    fail — the system cannot do its job
"""

from __future__ import annotations

import contextlib
import enum
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field

from app.core import hardware
from app.core.config import Config
from app.core.logging import get_logger
from app.core.paths import Paths
from app.core.proc import CommandError, RunOptions, run, which

log = get_logger(__name__)


class Status(str, enum.Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Check:
    key: str
    title: str
    status: Status
    detail: str = ""
    hint: str = ""
    category: str = "general"

    @property
    def passed(self) -> bool:
        return self.status is not Status.FAIL


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.WARN]

    def by_category(self) -> dict[str, list[Check]]:
        grouped: dict[str, list[Check]] = {}
        for check in self.checks:
            grouped.setdefault(check.category, []).append(check)
        return grouped

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "failures": len(self.failures),
            "warnings": len(self.warnings),
            "checks": [
                {
                    "key": c.key,
                    "title": c.title,
                    "status": c.status.value,
                    "detail": c.detail,
                    "hint": c.hint,
                    "category": c.category,
                }
                for c in self.checks
            ],
        }


# ffmpeg pieces the pipeline genuinely uses. Missing any of these means a
# feature silently does not work, so they are checked rather than assumed.
# Below this, generating a block competes with everything else on the box.
MIN_USABLE_CORES = 2

REQUIRED_FILTERS = ("xfade", "zoompan", "displace", "hue", "vignette", "noise", "fade")
REQUIRED_MUXERS = ("mpegts", "flv", "null")
REQUIRED_ENCODERS = ("libx264", "aac")

REQUIRED_PACKAGES = (
    ("fastapi", "the web dashboard"),
    ("uvicorn", "the web server"),
    ("sqlmodel", "the database layer"),
    ("pydantic", "configuration validation"),
    ("yaml", "reading config.yaml"),
    ("jinja2", "HTML templates"),
    ("numpy", "procedural visual generators"),
    ("PIL", "image handling"),
    ("mutagen", "reading music tags"),
    ("psutil", "system metrics"),
    ("bcrypt", "password hashing"),
    ("cryptography", "encrypting the stream key"),
)

OPTIONAL_PACKAGES = (
    ("numba", "JIT-compiled procedural generators (falls back to numpy)"),
    ("librosa", "BPM analysis at upload time"),
)


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def _check_binary(
    report: Report,
    binary: str,
    *,
    key: str,
    title: str,
    hint: str,
    required: bool,
    version_args: tuple[str, ...] = ("--version",),
) -> bool:
    path = which(binary)
    if not path:
        report.add(
            Check(
                key=key,
                title=title,
                status=Status.FAIL if required else Status.WARN,
                detail=f"{binary!r} is not on PATH",
                hint=hint,
                category="tools",
            )
        )
        return False

    version = ""
    try:
        result = run([binary, *version_args], RunOptions(timeout_s=20.0, check=False))
        first = (result.stdout or result.stderr).strip().splitlines()
        version = first[0] if first else ""
    except CommandError as exc:
        report.add(
            Check(
                key=key,
                title=title,
                status=Status.WARN,
                detail=f"found at {path} but did not report a version: {exc}",
                hint=hint,
                category="tools",
            )
        )
        return True

    report.add(
        Check(
            key=key,
            title=title,
            status=Status.OK,
            detail=f"{version} ({path})" if version else path,
            category="tools",
        )
    )
    return True


def check_tools(report: Report, cfg: Config) -> None:
    have_ffmpeg = _check_binary(
        report,
        cfg.tools.ffmpeg,
        key="ffmpeg",
        title="ffmpeg",
        hint="apt install ffmpeg — nothing encodes or streams without it",
        required=True,
        version_args=("-hide_banner", "-version"),
    )
    _check_binary(
        report,
        cfg.tools.ffprobe,
        key="ffprobe",
        title="ffprobe",
        hint="apt install ffmpeg — block validation needs it",
        required=True,
        version_args=("-hide_banner", "-version"),
    )
    _check_binary(
        report,
        cfg.audio.liquidsoap.binary,
        key="liquidsoap",
        title="Liquidsoap",
        hint="apt install liquidsoap — the audio engine",
        required=True,
    )
    if cfg.audio.icecast.managed:
        _check_binary(
            report,
            cfg.audio.icecast.binary,
            key="icecast",
            title="Icecast",
            hint="apt install icecast2 — carries audio from Liquidsoap to the streamer",
            required=True,
            version_args=("-v",),
        )

    if have_ffmpeg:
        _check_ffmpeg_features(report, cfg)


def _check_ffmpeg_features(report: Report, cfg: Config) -> None:
    encoders = set(hardware.ffmpeg_encoders(cfg.tools.ffmpeg))
    filters = set(hardware.ffmpeg_filters(cfg.tools.ffmpeg))
    muxers = set(hardware.ffmpeg_muxers(cfg.tools.ffmpeg))

    for label, needed, found, key in (
        ("encoders", REQUIRED_ENCODERS, encoders, "ffmpeg_encoders"),
        ("filters", REQUIRED_FILTERS, filters, "ffmpeg_filters"),
        ("muxers", REQUIRED_MUXERS, muxers, "ffmpeg_muxers"),
    ):
        missing = [name for name in needed if name not in found]
        if not found:
            report.add(
                Check(
                    key=key,
                    title=f"ffmpeg {label}",
                    status=Status.WARN,
                    detail=f"could not read the {label} list",
                    hint="ffmpeg answered unexpectedly; check it runs by hand",
                    category="tools",
                )
            )
        elif missing:
            report.add(
                Check(
                    key=key,
                    title=f"ffmpeg {label}",
                    status=Status.FAIL,
                    detail="missing: " + ", ".join(missing),
                    hint="this ffmpeg build is too limited; install the distribution package",
                    category="tools",
                )
            )
        else:
            report.add(
                Check(
                    key=key,
                    title=f"ffmpeg {label}",
                    status=Status.OK,
                    detail=f"all {len(needed)} required {label} present",
                    category="tools",
                )
            )


def check_python_packages(report: Report) -> None:
    import importlib.util

    missing = [
        (name, why) for name, why in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None
    ]
    if missing:
        report.add(
            Check(
                key="python_packages",
                title="Python packages",
                status=Status.FAIL,
                detail="missing: " + ", ".join(f"{n} ({w})" for n, w in missing),
                hint="pip install -r requirements.txt",
                category="runtime",
            )
        )
    else:
        report.add(
            Check(
                key="python_packages",
                title="Python packages",
                status=Status.OK,
                detail=f"all {len(REQUIRED_PACKAGES)} required packages import",
                category="runtime",
            )
        )

    absent = [
        (name, why) for name, why in OPTIONAL_PACKAGES if importlib.util.find_spec(name) is None
    ]
    if absent:
        report.add(
            Check(
                key="python_packages_optional",
                title="Optional packages",
                status=Status.WARN,
                detail="not installed: " + ", ".join(f"{n} ({w})" for n, w in absent),
                hint="optional; the system runs without them",
                category="runtime",
            )
        )


def check_paths(report: Report, paths: Paths, cfg: Config) -> None:
    for directory in paths.directories():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            report.add(
                Check(
                    key=f"dir:{directory.name}",
                    title=f"Directory {directory}",
                    status=Status.FAIL,
                    detail=str(exc),
                    hint="check ownership and permissions of the data directory",
                    category="storage",
                )
            )
            return

    try:
        with tempfile.NamedTemporaryFile(dir=paths.root, prefix=".preflight-", delete=True):
            pass
    except OSError as exc:
        report.add(
            Check(
                key="data_writable",
                title="Data directory is writable",
                status=Status.FAIL,
                detail=f"{paths.root}: {exc}",
                hint="in Docker, make sure the volume is writable by the container user",
                category="storage",
            )
        )
        return

    report.add(
        Check(
            key="data_writable",
            title="Data directory is writable",
            status=Status.OK,
            detail=str(paths.root),
            category="storage",
        )
    )
    _check_disk_space(report, paths, cfg)
    _check_fifo(report, paths)


def _check_disk_space(report: Report, paths: Paths, cfg: Config) -> None:
    usage = shutil.disk_usage(paths.root)
    free_gb = usage.free / 1024**3
    needed_gb = cfg.visual.pool.max_disk_gb
    if free_gb < needed_gb * 0.25:
        status, hint = Status.FAIL, (
            f"free up space or lower visual.pool.max_disk_gb (currently {needed_gb:.0f} GB)"
        )
    elif free_gb < needed_gb:
        status, hint = Status.WARN, (
            f"the pool is allowed {needed_gb:.0f} GB but only {free_gb:.1f} GB is free"
        )
    else:
        status, hint = Status.OK, ""
    report.add(
        Check(
            key="disk_space",
            title="Free disk space",
            status=status,
            detail=f"{free_gb:.1f} GB free of {usage.total / 1024**3:.1f} GB",
            hint=hint,
            category="storage",
        )
    )


def _check_fifo(report: Report, paths: Paths) -> None:
    """The streamer feeds ffmpeg through a named pipe; prove one can exist."""
    probe = paths.state / ".preflight.fifo"
    try:
        if probe.exists():
            probe.unlink()
        os.mkfifo(probe, 0o600)
    except (OSError, AttributeError) as exc:
        report.add(
            Check(
                key="fifo",
                title="Named pipes",
                status=Status.FAIL,
                detail=str(exc),
                hint=(
                    "the state directory must be on a filesystem that supports FIFOs "
                    "(not a Windows bind mount, not FAT/NTFS)"
                ),
                category="storage",
            )
        )
        return
    finally:
        with contextlib.suppress(OSError):
            probe.unlink(missing_ok=True)
    report.add(
        Check(
            key="fifo",
            title="Named pipes",
            status=Status.OK,
            detail="the state directory supports FIFOs",
            category="storage",
        )
    )


def check_database(report: Report, paths: Paths) -> None:
    try:
        connection = sqlite3.connect(paths.db_file)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            version = sqlite3.sqlite_version
        finally:
            connection.close()
    except sqlite3.Error as exc:
        report.add(
            Check(
                key="database",
                title="SQLite database",
                status=Status.FAIL,
                detail=str(exc),
                hint=f"cannot open {paths.db_file}",
                category="storage",
            )
        )
        return
    report.add(
        Check(
            key="database",
            title="SQLite database",
            status=Status.OK,
            detail=f"SQLite {version} at {paths.db_file}",
            category="storage",
        )
    )


def check_hardware(report: Report, cfg: Config) -> None:
    cpu = hardware.cpu_info()
    report.add(
        Check(
            key="cpu",
            title="CPU",
            status=Status.OK if cpu.count >= MIN_USABLE_CORES else Status.WARN,
            detail=f"{cpu.count} usable cores — {cpu.model}",
            hint=(
                ""
                if cpu.count >= MIN_USABLE_CORES
                else "with a single core, generation will be very slow"
            ),
            category="hardware",
        )
    )

    gpu = hardware.gpu_info()
    report.add(
        Check(
            key="gpu",
            title="GPU",
            status=Status.OK,
            detail=(
                f"{gpu.name}, driver {gpu.driver}, {gpu.memory_mb} MB"
                if gpu.present
                else "none detected — CPU-only mode"
            ),
            category="hardware",
        )
    )

    encoder, why = hardware.resolve_encoder(cfg.video.encoder, ffmpeg=cfg.tools.ffmpeg)
    report.add(
        Check(
            key="encoder",
            title="Video encoder",
            status=Status.OK,
            detail=f"{encoder} ({why})",
            category="hardware",
        )
    )

    backend, why = hardware.resolve_ai_backend(cfg.visual.ai.backend)
    if backend == "none":
        status = Status.WARN if cfg.visual.sources.ai_ratio > 0 else Status.OK
        hint = (
            "visual.sources.ai_ratio is above zero but no AI backend is available; "
            "blocks will be procedural only"
            if cfg.visual.sources.ai_ratio > 0
            else "procedural visuals need no model, so this is fine"
        )
    else:
        status, hint = Status.OK, ""
    report.add(
        Check(
            key="ai_backend",
            title="AI image backend",
            status=status,
            detail=f"{backend} ({why})",
            hint=hint,
            category="hardware",
        )
    )


def check_config(report: Report, cfg: Config, paths: Paths) -> None:
    report.add(
        Check(
            key="config_file",
            title="Configuration",
            status=Status.OK,
            detail=(
                f"loaded from {cfg.source_path}"
                if cfg.source_path
                else "no config.yaml found; using defaults plus TAD_* environment"
            ),
            hint=(
                ""
                if cfg.source_path
                else "copy config/config.example.yaml to config/config.yaml to customise"
            ),
            category="config",
        )
    )

    if cfg.auth.enabled and not cfg.auth.password_hash:
        report.add(
            Check(
                key="auth",
                title="Dashboard password",
                status=Status.WARN,
                detail="no password is set",
                hint="the dashboard will ask you to choose one on first visit",
                category="config",
            )
        )
    elif not cfg.auth.enabled:
        report.add(
            Check(
                key="auth",
                title="Dashboard password",
                status=Status.WARN,
                detail="authentication is disabled",
                hint="anyone who can reach the port controls the stream",
                category="config",
            )
        )
    else:
        report.add(
            Check(
                key="auth",
                title="Dashboard password",
                status=Status.OK,
                detail="a password hash is configured",
                category="config",
            )
        )

    _check_broadcast_title(report, cfg, paths)

    if cfg.stream.video_mode == "reencode":
        report.add(
            Check(
                key="video_mode",
                title="Video mode",
                status=Status.WARN,
                detail="reencode — the streamer will re-encode video continuously",
                hint="'copy' is the low-CPU design; use reencode only if joins misbehave",
                category="config",
            )
        )
    else:
        report.add(
            Check(
                key="video_mode",
                title="Video mode",
                status=Status.OK,
                detail="copy — video is never re-encoded while streaming",
                category="config",
            )
        )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run_preflight(cfg: Config, paths: Paths | None = None) -> Report:
    paths = paths or Paths.from_config(cfg)
    report = Report()
    check_config(report, cfg, paths)
    check_python_packages(report)
    check_tools(report, cfg)
    check_paths(report, paths, cfg)
    check_database(report, paths)
    check_hardware(report, cfg)
    return report


def format_report(report: Report, *, color: bool = True) -> str:
    symbols = {Status.OK: "✔", Status.WARN: "▲", Status.FAIL: "✖"}
    colors = {Status.OK: "\033[32m", Status.WARN: "\033[33m", Status.FAIL: "\033[31m"}
    reset = "\033[0m" if color else ""

    lines: list[str] = []
    for category, checks in report.by_category().items():
        lines.append(f"\n{category.upper()}")
        for check in checks:
            prefix = colors[check.status] if color else ""
            lines.append(f"  {prefix}{symbols[check.status]}{reset} {check.title}: {check.detail}")
            if check.hint and check.status is not Status.OK:
                lines.append(f"      → {check.hint}")

    lines.append("")
    if report.ok:
        verdict = f"PASS — {len(report.warnings)} warning(s)"
        prefix = colors[Status.OK] if color else ""
    else:
        verdict = f"FAIL — {len(report.failures)} blocking problem(s)"
        prefix = colors[Status.FAIL] if color else ""
    lines.append(f"{prefix}{verdict}{reset}")
    return "\n".join(lines)


__all__ = ["Check", "Report", "Status", "format_report", "run_preflight"]


def _check_broadcast_title(report: Report, cfg: Config, paths: Paths) -> None:
    """A title nobody can deliver is worse than no title at all.

    RTMP carries video and audio and nothing else: the name of a broadcast
    lives on the YouTube side and is only reachable through the Data API. So a
    title set without a connected account is saved, shown back, and then
    quietly ignored every time the station goes live.
    """
    template = cfg.stream.youtube.title_template
    if not template:
        return

    connected = False
    with contextlib.suppress(Exception):
        from app.core.runtime import SECRET_YT_CLIENT_ID, SECRET_YT_REFRESH_TOKEN
        from app.core.security import SecretStore

        store = SecretStore.open(paths.secrets_file)
        connected = bool(store.get(SECRET_YT_CLIENT_ID) and store.get(SECRET_YT_REFRESH_TOKEN))

    report.add(
        Check(
            key="broadcast_title",
            title="Broadcast title",
            status=Status.OK if connected else Status.WARN,
            detail=(
                f"{template!r} will be applied when the broadcast starts"
                if connected
                else f"{template!r} is set but no YouTube account is connected"
            ),
            hint=(
                ""
                if connected
                else "the title cannot travel over RTMP; connect an account on the "
                "stream page or broadcasts keep the name YouTube gives them"
            ),
            category="config",
        )
    )
