"""OpenVINO, for Intel CPUs.

The brief named this the CPU default. On the AMD machine this was developed on
it still runs — the CPU plugin is generic x86 — but it is not obviously better
than stable-diffusion.cpp there, so ``auto`` prefers sd.cpp and only reaches
for OpenVINO when it is explicitly asked for or nothing else is present.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.config import Config
from app.core.hardware import cpu_count
from app.core.logging import get_logger
from app.services.visual.ai.base import (
    BackendInfo,
    BackendUnavailable,
    GeneratedImage,
    ImageBackend,
    register,
)

log = get_logger(__name__)


@register
class OpenVinoBackend(ImageBackend):
    name = "openvino"

    def __init__(self, cfg: Config, models_dir: Path) -> None:
        super().__init__(cfg, models_dir)
        self._pipe = None

    @classmethod
    def _model_dir(cls, cfg: Config, models_dir: Path) -> Path:
        return models_dir / "openvino" / cfg.visual.ai.model_id.replace("/", "--")

    @classmethod
    def probe(cls, cfg: Config, models_dir: Path) -> BackendInfo:
        try:
            import openvino  # noqa: F401
        except ImportError:
            return BackendInfo(cls.name, False, "openvino is not installed")
        try:
            from optimum.intel import OVStableDiffusionPipeline  # noqa: F401
        except ImportError:
            return BackendInfo(cls.name, False, "optimum-intel is not installed")

        target = cls._model_dir(cfg, models_dir)
        if not target.is_dir():
            return BackendInfo(
                cls.name,
                False,
                f"no converted model at {target} — run scripts/fetch_models.py --backend openvino",
                cfg.visual.ai.model_id,
                "cpu",
            )
        return BackendInfo(
            cls.name, True, "openvino runtime and model present", cfg.visual.ai.model_id, "cpu"
        )

    def load(self) -> None:
        if self._loaded:
            return
        info = self.probe(self.cfg, self.models_dir)
        if not info.available:
            raise BackendUnavailable(info.reason)

        from optimum.intel import OVStableDiffusionPipeline

        threads = self.cfg.visual.ai.threads or max(1, cpu_count() - 2)
        log.info("loading the OpenVINO model", extra={"threads": threads})
        started = time.monotonic()
        self._pipe = OVStableDiffusionPipeline.from_pretrained(
            str(self._model_dir(self.cfg, self.models_dir)),
            compile=False,
            ov_config={"INFERENCE_NUM_THREADS": str(threads)},
        )
        self._pipe.compile()
        self._loaded = True
        log.info("model loaded in %.1fs", time.monotonic() - started)

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False

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
        if not self._loaded:
            self.load()

        import numpy as np

        ai = self.cfg.visual.ai
        steps = steps or ai.steps
        width = width or ai.width
        height = height or ai.height

        started = time.monotonic()
        result = self._pipe(  # type: ignore[misc]
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=ai.guidance_scale,
            width=width,
            height=height,
            generator=np.random.RandomState(seed),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out_path)

        image = GeneratedImage(
            path=out_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            backend=self.name,
            model_id=ai.model_id,
            seconds=time.monotonic() - started,
        )
        image.write_sidecar()
        return image
