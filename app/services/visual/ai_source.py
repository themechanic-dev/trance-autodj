"""Building clips from AI stills.

Sits between the backend (which makes pictures) and the animator (which makes
video). Its whole job is to fail softly: a model that will not load, a prompt
that produces nothing, a backend that disappears — none of them may stop the
generator, because the procedural sources are always there.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Config
from app.core.logging import get_logger
from app.core.paths import Paths
from app.services.visual import ai as ai_backends
from app.services.visual.ai.base import BackendUnavailable, GeneratedImage, ImageBackend
from app.services.visual.ai.prompts import PromptSet
from app.services.visual.stills import StillAnimator, load_image

log = get_logger(__name__)


@dataclass
class AiSource:
    """A loaded backend plus its prompts. None means "procedural only"."""

    backend: ImageBackend | None
    reason: str
    prompts: PromptSet

    @property
    def available(self) -> bool:
        return self.backend is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "backend": self.backend.name if self.backend else "none",
            "reason": self.reason,
            "presets": len(self.prompts),
        }


def build(cfg: Config, paths: Paths) -> AiSource:
    backend, reason = ai_backends.resolve(cfg, paths.models)
    prompts = PromptSet.load(Path(cfg.app.config_dir) / "prompts.json")
    if backend is None:
        log.info("no AI image backend: %s", reason)
    else:
        log.info("AI image backend ready: %s", reason)
    return AiSource(backend=backend, reason=reason, prompts=prompts)


def generate_images(
    source: AiSource,
    *,
    cfg: Config,
    paths: Paths,
    count: int,
    rng: random.Random,
) -> list[GeneratedImage]:
    """Make up to ``count`` images. Returns fewer, or none, on failure."""
    if source.backend is None:
        return []

    images: list[GeneratedImage] = []
    for index in range(count):
        preset = source.prompts.choose(rng)
        seed = rng.randrange(1, 2**31)
        target = paths.images / f"img_{uuid.uuid4().hex}.png"
        try:
            image = source.backend.generate(
                preset.full_prompt(source.prompts.style_suffix),
                negative_prompt=source.prompts.negative_prompt,
                seed=seed,
                out_path=target,
            )
        except (BackendUnavailable, OSError, RuntimeError) as exc:
            # One failed image is not a failed block: carry on with what we
            # have, and let the caller fall back if that is nothing.
            log.warning("image %d/%d failed: %s", index + 1, count, exc)
            target.unlink(missing_ok=True)
            continue
        image.preset_id = preset.id
        images.append(image)
        log.info(
            "image generated",
            extra={"preset": preset.id, "seconds": round(image.seconds, 1), "seed": seed},
        )
    return images


def animator_factory(
    images: list[GeneratedImage],
    *,
    width: int,
    height: int,
):
    """A factory the animator can call repeatedly, with the pixels preloaded.

    Decoding the PNGs once here rather than per pass matters: forcing a loop
    renders the clip twice, and re-reading the files would double the I/O for
    no reason.
    """
    loaded: list[np.ndarray] = []
    metadata: list[dict] = []
    for image in images:
        try:
            loaded.append(load_image(image.path, width, height))
            metadata.append(image.as_dict())
        except (OSError, ValueError) as exc:
            log.warning("could not load %s: %s", image.path.name, exc)

    if not loaded:
        return None

    def factory() -> StillAnimator:
        return StillAnimator(loaded, metadata=metadata)

    return factory


def cleanup(images: list[GeneratedImage], *, keep: bool = False) -> None:
    """Remove the stills once their clip exists, unless asked to keep them."""
    if keep:
        return
    for image in images:
        image.path.unlink(missing_ok=True)
        image.path.with_suffix(".json").unlink(missing_ok=True)


__all__ = ["AiSource", "animator_factory", "build", "cleanup", "generate_images"]
