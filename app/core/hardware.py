"""Capability detection: what this machine can actually do, right now.

The same image runs on a GPU workstation and on a headless CPU-only VM, so
nothing is assumed. Every capability is *probed*, and the expensive probes
(which run a real encode) are cached for the life of the process.

The two questions this module answers:

    encoder   -> "libx264" or "h264_nvenc"
    ai backend-> "cuda", "sdcpp", "openvino" or "none"

A listed encoder is not a working encoder: ffmpeg advertises h264_nvenc
whenever it was built with support, including on machines with no NVIDIA card
at all. So the NVENC probe encodes one real frame and checks the exit code.
"""

from __future__ import annotations

import functools
import os
import platform
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.core.proc import CommandError, RunOptions, run, which

log = get_logger(__name__)

PROBE_TIMEOUT_S = 30.0

# Column positions in the outputs we parse.
_SMI_DRIVER, _SMI_MEMORY = 1, 2  # nvidia-smi --query-gpu=name,driver,memory
_FFMPEG_VERSION_WORD = 2  # "ffmpeg version <v> Copyright ..."
_LISTING_NAME_COLUMN = 1  # "<flags> <name> <description>"


@dataclass(frozen=True)
class GpuInfo:
    present: bool
    name: str = ""
    driver: str = ""
    memory_mb: int = 0
    cuda_version: str = ""


@dataclass(frozen=True)
class CpuInfo:
    count: int
    model: str
    architecture: str


@dataclass(frozen=True)
class Capabilities:
    cpu: CpuInfo
    gpu: GpuInfo
    ffmpeg_path: str = ""
    ffmpeg_version: str = ""
    nvenc_available: bool = False
    nvenc_reason: str = ""
    torch_cuda: bool = False
    openvino_available: bool = False
    sdcpp_path: str = ""
    encoders: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------


def cpu_count() -> int:
    """Cores this process may actually use.

    ``os.cpu_count()`` reports the host's cores even inside a container that
    has been limited to two of them, which is exactly how a "low CPU" design
    ends up oversubscribing. cgroup v2 first, then affinity, then the host.
    """
    quota_file = Path("/sys/fs/cgroup/cpu.max")
    if quota_file.is_file():
        try:
            quota_raw, period_raw = quota_file.read_text().split()
            if quota_raw != "max":
                quota, period = int(quota_raw), int(period_raw)
                if period > 0:
                    limited = max(1, int(quota / period))
                    return min(limited, os.cpu_count() or limited)
        except (OSError, ValueError):
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux
        return max(1, os.cpu_count() or 1)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def cpu_info() -> CpuInfo:
    return CpuInfo(count=cpu_count(), model=_cpu_model(), architecture=platform.machine())


def load_average() -> tuple[float, float, float]:
    try:
        return os.getloadavg()
    except (OSError, AttributeError):  # pragma: no cover - non-Linux
        return (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def gpu_info() -> GpuInfo:
    """Ask nvidia-smi. Absent tool or non-zero exit means "no usable GPU"."""
    if not which("nvidia-smi"):
        return GpuInfo(present=False)
    try:
        result = run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            RunOptions(timeout_s=PROBE_TIMEOUT_S, check=True),
        )
    except CommandError as exc:
        log.debug("nvidia-smi failed, treating as no GPU: %s", exc)
        return GpuInfo(present=False)

    first = next((ln for ln in result.stdout.splitlines() if ln.strip()), "")
    if not first:
        return GpuInfo(present=False)
    fields = [part.strip() for part in first.split(",")]
    name = fields[0] if fields else ""
    driver = fields[_SMI_DRIVER] if len(fields) > _SMI_DRIVER else ""
    try:
        memory_mb = int(float(fields[_SMI_MEMORY])) if len(fields) > _SMI_MEMORY else 0
    except ValueError:
        memory_mb = 0
    return GpuInfo(present=True, name=name, driver=driver, memory_mb=memory_mb)


@functools.lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception as exc:
        log.warning("torch is installed but CUDA is unusable: %s", exc)
        return False


@functools.lru_cache(maxsize=1)
def openvino_available() -> bool:
    try:
        import openvino  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def ffmpeg_version(binary: str = "ffmpeg") -> str:
    if not which(binary):
        return ""
    try:
        result = run([binary, "-hide_banner", "-version"], RunOptions(timeout_s=PROBE_TIMEOUT_S))
    except CommandError:
        return ""
    first = result.stdout.splitlines()[0] if result.stdout else ""
    # "ffmpeg version 8.0.1-3ubuntu2 Copyright ..." -> "8.0.1-3ubuntu2"
    parts = first.split()
    return parts[_FFMPEG_VERSION_WORD] if len(parts) > _FFMPEG_VERSION_WORD else first


# A legend line such as "  T.. = Timeline support" or " V..... = Video".
_LEGEND_LINE = re.compile(r"^\s*[A-Za-z.|]{1,8}\s*=\s")


def _list_ffmpeg(binary: str, kind: str) -> tuple[str, ...]:
    """Names from ``ffmpeg -encoders`` / ``-filters`` / ``-muxers``.

    The three listings look similar but are not identical, and assuming they
    are is how this silently returns nothing:

        Encoders:  legend, then a "------" rule, then " V....D libx264  ..."
        Formats:   legend, then a "---" rule,    then "  E  mpegts     ..."
        Filters:   legend, NO rule at all,       then " .S xfade       ..."

    The flag column is also a different width in each (6, 3 and 2 characters).
    What they do share: a one-word header, legend lines with an early "=",
    optional dashes, and then entries whose *second* whitespace-separated
    column is the name. So that is what this keys on.
    """
    if not which(binary):
        return ()
    try:
        result = run([binary, "-hide_banner", f"-{kind}"], RunOptions(timeout_s=PROBE_TIMEOUT_S))
    except CommandError:
        return ()

    names: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped) == {"-"}:  # the separator rule, where there is one
            continue
        if _LEGEND_LINE.match(line):
            continue
        parts = line.split()
        if len(parts) > _LISTING_NAME_COLUMN:
            names.append(parts[_LISTING_NAME_COLUMN])
    return tuple(sorted(set(names)))


@functools.lru_cache(maxsize=1)
def ffmpeg_encoders(binary: str = "ffmpeg") -> tuple[str, ...]:
    return _list_ffmpeg(binary, "encoders")


@functools.lru_cache(maxsize=1)
def ffmpeg_filters(binary: str = "ffmpeg") -> tuple[str, ...]:
    return _list_ffmpeg(binary, "filters")


@functools.lru_cache(maxsize=1)
def ffmpeg_muxers(binary: str = "ffmpeg") -> tuple[str, ...]:
    return _list_ffmpeg(binary, "muxers")


@functools.lru_cache(maxsize=1)
def nvenc_usable(binary: str = "ffmpeg") -> tuple[bool, str]:
    """Encode one frame with h264_nvenc and see whether it survives.

    Returns (usable, reason). The reason is kept for the dashboard so the
    answer to "why is it using x264?" is on screen instead of in a log.
    """
    if not which(binary):
        return False, "ffmpeg not found"
    if not gpu_info().present:
        return False, "no NVIDIA GPU detected"
    if "h264_nvenc" not in ffmpeg_encoders(binary):
        return False, "this ffmpeg build has no h264_nvenc encoder"
    try:
        run(
            [
                binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.1:r=10",
                "-c:v",
                "h264_nvenc",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            RunOptions(timeout_s=PROBE_TIMEOUT_S, check=True),
        )
    except CommandError as exc:
        reason = (exc.stderr or str(exc)).strip().splitlines()
        detail = reason[-1] if reason else "probe encode failed"
        return False, detail[:200]
    return True, "verified by a test encode"


# --------------------------------------------------------------------------
# resolution of "auto" settings
# --------------------------------------------------------------------------


def resolve_encoder(setting: str, *, ffmpeg: str = "ffmpeg") -> tuple[str, str]:
    """Turn ``video.encoder`` into a concrete encoder plus an explanation."""
    if setting == "libx264":
        return "libx264", "set explicitly in config"
    if setting == "h264_nvenc":
        usable, reason = nvenc_usable(ffmpeg)
        if usable:
            return "h264_nvenc", "set explicitly in config"
        log.warning("h264_nvenc was requested but is not usable (%s); using libx264", reason)
        return "libx264", f"h264_nvenc requested but unusable: {reason}"
    usable, reason = nvenc_usable(ffmpeg)
    if usable:
        return "h264_nvenc", f"auto: {reason}"
    return "libx264", f"auto: {reason}"


def resolve_ai_backend(setting: str, *, sdcpp_binary: str = "sd") -> tuple[str, str]:
    """Turn ``visual.ai.backend`` into a concrete backend plus an explanation.

    Order for ``auto``: CUDA (fastest by a wide margin), then
    stable-diffusion.cpp (works anywhere, no PyTorch), then OpenVINO (Intel
    CPUs), then none. "none" is a perfectly good answer: procedural visuals
    need no model at all.
    """
    if setting == "none":
        return "none", "disabled in config"

    def _cuda() -> tuple[bool, str]:
        if not gpu_info().present:
            return False, "no NVIDIA GPU detected"
        if not torch_cuda_available():
            return False, "torch with CUDA support is not installed"
        return True, f"CUDA on {gpu_info().name}"

    def _sdcpp() -> tuple[bool, str]:
        found = shutil.which(sdcpp_binary)
        return (
            (True, f"stable-diffusion.cpp at {found}")
            if found
            else (
                False,
                f"{sdcpp_binary!r} not found on PATH",
            )
        )

    def _openvino() -> tuple[bool, str]:
        return (
            (True, "openvino runtime present")
            if openvino_available()
            else (
                False,
                "openvino is not installed",
            )
        )

    checks = {"cuda": _cuda, "sdcpp": _sdcpp, "openvino": _openvino}

    if setting in checks:
        ok, reason = checks[setting]()
        if ok:
            return setting, reason
        log.warning("AI backend %r was requested but is unavailable: %s", setting, reason)
        return "none", f"{setting} requested but unavailable: {reason}"

    reasons: list[str] = []
    for name in ("cuda", "sdcpp", "openvino"):
        ok, reason = checks[name]()
        if ok:
            return name, f"auto: {reason}"
        reasons.append(f"{name}: {reason}")
    return "none", "auto: no AI backend available (" + "; ".join(reasons) + ")"


@dataclass
class _Detected:
    value: Capabilities | None = None


_cache = _Detected()


def detect(*, ffmpeg: str = "ffmpeg", refresh: bool = False) -> Capabilities:
    """Full capability sweep. Cached; pass ``refresh=True`` to re-probe."""
    if _cache.value is not None and not refresh:
        return _cache.value
    if refresh:
        for fn in (
            gpu_info,
            torch_cuda_available,
            openvino_available,
            ffmpeg_version,
            ffmpeg_encoders,
            ffmpeg_filters,
            ffmpeg_muxers,
            nvenc_usable,
        ):
            fn.cache_clear()

    nvenc_ok, nvenc_reason = nvenc_usable(ffmpeg)
    gpu = gpu_info()
    caps = Capabilities(
        cpu=cpu_info(),
        gpu=gpu,
        ffmpeg_path=which(ffmpeg) or "",
        ffmpeg_version=ffmpeg_version(ffmpeg),
        nvenc_available=nvenc_ok,
        nvenc_reason=nvenc_reason,
        torch_cuda=torch_cuda_available(),
        openvino_available=openvino_available(),
        sdcpp_path=shutil.which("sd") or "",
        encoders=ffmpeg_encoders(ffmpeg),
        filters=ffmpeg_filters(ffmpeg),
    )
    _cache.value = caps
    return caps


__all__ = [
    "Capabilities",
    "CpuInfo",
    "GpuInfo",
    "cpu_count",
    "cpu_info",
    "detect",
    "ffmpeg_encoders",
    "ffmpeg_filters",
    "ffmpeg_muxers",
    "ffmpeg_version",
    "gpu_info",
    "load_average",
    "nvenc_usable",
    "resolve_ai_backend",
    "resolve_encoder",
]
