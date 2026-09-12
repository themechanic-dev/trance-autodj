"""AI image backends.

Importing this package registers every backend, so ``available()`` is complete
after one import.
"""

from pathlib import Path

from app.core.config import Config
from app.core.logging import get_logger

# Imported for their registration side effect.
from app.services.visual.ai import openvino_backend as _openvino  # noqa: F401
from app.services.visual.ai import sdcpp_backend as _sdcpp  # noqa: F401
from app.services.visual.ai import torch_backend as _torch  # noqa: F401
from app.services.visual.ai.base import (
    BackendInfo,
    BackendUnavailable,
    GeneratedImage,
    ImageBackend,
    available,
    get,
    register,
)
from app.services.visual.ai.prompts import Preset, PromptSet

log = get_logger(__name__)

#: Order tried when visual.ai.backend is "auto". CUDA first because it is an
#: order of magnitude faster; then sd.cpp, which runs anywhere; then OpenVINO.
AUTO_ORDER = ("cuda", "sdcpp", "openvino")


def probe_all(cfg: Config, models_dir: Path) -> list[BackendInfo]:
    """Ask every backend whether it can run. Never raises."""
    results: list[BackendInfo] = []
    for name, cls in available().items():
        try:
            results.append(cls.probe(cfg, models_dir))
        except Exception as exc:
            results.append(BackendInfo(name, False, f"probe failed: {exc}"))
    return sorted(results, key=lambda info: (not info.available, info.name))


def resolve(cfg: Config, models_dir: Path) -> tuple[ImageBackend | None, str]:
    """Pick a usable backend. Returns (backend or None, explanation).

    Returning None is a perfectly good outcome: procedural generators need no
    model, so the station keeps producing visuals either way.
    """
    setting = cfg.visual.ai.backend
    if setting == "none":
        return None, "disabled in configuration"

    order = AUTO_ORDER if setting == "auto" else (setting,)
    reasons: list[str] = []
    for name in order:
        try:
            cls = get(name)
        except BackendUnavailable as exc:
            reasons.append(str(exc))
            continue
        info = cls.probe(cfg, models_dir)
        if info.available:
            prefix = "auto: " if setting == "auto" else ""
            return cls(cfg, models_dir), f"{prefix}{name} — {info.reason}"
        reasons.append(f"{name}: {info.reason}")

    return None, "no AI backend available (" + "; ".join(reasons) + ")"


__all__ = [
    "AUTO_ORDER",
    "BackendInfo",
    "BackendUnavailable",
    "GeneratedImage",
    "ImageBackend",
    "Preset",
    "PromptSet",
    "available",
    "get",
    "probe_all",
    "register",
    "resolve",
]
