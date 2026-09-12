"""The AI image backend interface.

Backends differ enormously in how they run — a PyTorch pipeline in-process, an
OpenVINO graph, a C++ binary — but they all answer the same question: given a
prompt and a seed, produce an image. Everything above this line only knows
that.

A backend that cannot load is not an error. The visual generator falls back to
procedural sources, which need no model at all, and the dashboard says why.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Config
from app.core.logging import get_logger

log = get_logger(__name__)


class BackendUnavailable(RuntimeError):  # noqa: N818 - a state, not a crash
    """Raised when a backend cannot be used, with a reason a human can act on."""


@dataclass
class GeneratedImage:
    path: Path
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    width: int
    height: int
    backend: str
    model_id: str
    seconds: float
    preset_id: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "file": self.path.name,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "steps": self.steps,
            "size": f"{self.width}x{self.height}",
            "backend": self.backend,
            "model": self.model_id,
            "seconds": round(self.seconds, 1),
            "preset": self.preset_id,
        }

    def write_sidecar(self) -> Path:
        """Provenance next to the image: prompt, seed, model, when.

        Enough to reproduce the picture, which matters when one turns out
        well and you want more like it.
        """
        sidecar = self.path.with_suffix(".json")
        payload = {
            **self.as_dict(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return sidecar


@dataclass
class BackendInfo:
    name: str
    available: bool
    reason: str
    model_id: str = ""
    device: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
            "model": self.model_id,
            "device": self.device,
        }


class ImageBackend(ABC):
    """One way of turning a prompt into an image."""

    name: str = "unnamed"

    def __init__(self, cfg: Config, models_dir: Path) -> None:
        self.cfg = cfg
        self.models_dir = models_dir
        self._loaded = False

    # -- availability ------------------------------------------------------

    @classmethod
    @abstractmethod
    def probe(cls, cfg: Config, models_dir: Path) -> BackendInfo:
        """Can this backend run here, and if not, why not?

        Must be cheap and must never raise: it is called on every dashboard
        refresh and during preflight.
        """

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Bring the model into memory. May take minutes. Raises BackendUnavailable."""

    def unload(self) -> None:
        """Release the model. Called when generation is paused for a long time."""
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # -- work --------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str,
        seed: int,
        out_path: Path,
        steps: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> GeneratedImage:
        """Produce one image at ``out_path``."""


_REGISTRY: dict[str, type[ImageBackend]] = {}


def register(cls: type[ImageBackend]) -> type[ImageBackend]:
    _REGISTRY[cls.name] = cls
    return cls


def available() -> dict[str, type[ImageBackend]]:
    return dict(_REGISTRY)


def get(name: str) -> type[ImageBackend]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise BackendUnavailable(
            f"unknown AI backend {name!r}; known: {', '.join(sorted(_REGISTRY))}"
        ) from None


__all__ = [
    "BackendInfo",
    "BackendUnavailable",
    "GeneratedImage",
    "ImageBackend",
    "available",
    "get",
    "register",
]
