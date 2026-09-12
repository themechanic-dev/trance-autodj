"""Diffusers, on whatever torch device is available.

Called "cuda" in configuration because that is the case worth having: an
SD-Turbo step takes well under a second on a modern NVIDIA card and 15-40
seconds on a CPU. The class still runs on CPU, which makes it testable without
a GPU, but ``auto`` selection only picks it when CUDA is actually usable.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.config import Config
from app.core.hardware import gpu_info, torch_cuda_available
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
class TorchBackend(ImageBackend):
    name = "cuda"

    def __init__(self, cfg: Config, models_dir: Path) -> None:
        super().__init__(cfg, models_dir)
        self._pipe = None
        self._device = "cpu"

    # -- availability ------------------------------------------------------

    @classmethod
    def probe(cls, cfg: Config, models_dir: Path) -> BackendInfo:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return BackendInfo(
                cls.name,
                False,
                "torch is not installed — build the image with --build-arg WITH_CUDA=1",
            )
        if importlib.util.find_spec("diffusers") is None:
            return BackendInfo(cls.name, False, "diffusers is not installed")

        if not torch_cuda_available():
            gpu = gpu_info()
            reason = (
                f"{gpu.name} is present but torch cannot use CUDA"
                if gpu.present
                else "no NVIDIA GPU detected"
            )
            return BackendInfo(cls.name, False, reason, cfg.visual.ai.model_id, "cpu")

        if not cls._model_present(cfg, models_dir):
            return BackendInfo(
                cls.name,
                False,
                "the model is not downloaded — run scripts/fetch_models.py",
                cfg.visual.ai.model_id,
                "cuda",
            )
        return BackendInfo(
            cls.name, True, f"CUDA on {gpu_info().name}", cfg.visual.ai.model_id, "cuda"
        )

    @staticmethod
    def _model_present(cfg: Config, models_dir: Path) -> bool:
        """Has the model been fetched into the local cache?

        Two layouts are accepted: a plain directory, and the ``models--org--name``
        shape the HuggingFace cache uses.
        """
        slug = cfg.visual.ai.model_id.replace("/", "--")
        return (models_dir / slug).is_dir() or (models_dir / f"models--{slug}").is_dir()

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        info = self.probe(self.cfg, self.models_dir)
        if not info.available:
            raise BackendUnavailable(info.reason)

        import torch
        from diffusers import AutoPipelineForText2Image

        self._device = "cuda" if torch_cuda_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        log.info(
            "loading the diffusion model",
            extra={"model": self.cfg.visual.ai.model_id, "device": self._device},
        )
        started = time.monotonic()
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.cfg.visual.ai.model_id,
            torch_dtype=dtype,
            cache_dir=str(self.models_dir),
            local_files_only=True,  # never reach for the network mid-broadcast
            safety_checker=None,
        )
        self._pipe = self._pipe.to(self._device)
        if self._device == "cuda":
            self._pipe.enable_attention_slicing()
        else:
            threads = self.cfg.visual.ai.threads or max(1, (torch.get_num_threads() or 2) - 2)
            torch.set_num_threads(threads)

        self._loaded = True
        log.info("model loaded in %.1fs", time.monotonic() - started)

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False
        try:
            import torch

            if torch_cuda_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- work --------------------------------------------------------------

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

        import torch

        ai = self.cfg.visual.ai
        steps = steps or ai.steps
        width = width or ai.width
        height = height or ai.height

        started = time.monotonic()
        generator = torch.Generator(device=self._device).manual_seed(seed)
        result = self._pipe(  # type: ignore[misc]
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=ai.guidance_scale,
            width=width,
            height=height,
            generator=generator,
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
