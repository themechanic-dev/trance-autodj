"""stable-diffusion.cpp — a single binary, quantised weights, no PyTorch.

The most portable of the three: it needs no Python ML stack at all, runs on
AMD and ARM as happily as Intel, and a q4 GGUF of SD-Turbo is under 2 GB. That
makes it the sensible default for a CPU-only box, which is what this system is
usually deployed on.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from app.core.config import Config
from app.core.hardware import cpu_count
from app.core.logging import get_logger
from app.core.proc import CommandError, RunOptions, run
from app.services.visual.ai.base import (
    BackendInfo,
    BackendUnavailable,
    GeneratedImage,
    ImageBackend,
    register,
)

log = get_logger(__name__)

BINARY_NAMES = ("sd", "stable-diffusion", "sd-cli")
MODEL_SUFFIXES = (".gguf", ".safetensors", ".ckpt")


@register
class StableDiffusionCppBackend(ImageBackend):
    name = "sdcpp"

    @classmethod
    def _binary(cls) -> str | None:
        for candidate in BINARY_NAMES:
            found = shutil.which(candidate)
            if found:
                return found
        return None

    @classmethod
    def _model_file(cls, models_dir: Path) -> Path | None:
        directory = models_dir / "sdcpp"
        if not directory.is_dir():
            return None
        for suffix in MODEL_SUFFIXES:
            files = sorted(directory.glob(f"*{suffix}"))
            if files:
                return files[0]
        return None

    @classmethod
    def probe(cls, cfg: Config, models_dir: Path) -> BackendInfo:
        binary = cls._binary()
        if not binary:
            return BackendInfo(
                cls.name,
                False,
                f"none of {', '.join(BINARY_NAMES)} is on PATH — build stable-diffusion.cpp",
            )
        model = cls._model_file(models_dir)
        if model is None:
            return BackendInfo(
                cls.name,
                False,
                f"no model file in {models_dir / 'sdcpp'} — "
                "run scripts/fetch_models.py --backend sdcpp",
                device="cpu",
            )
        return BackendInfo(cls.name, True, f"{binary} with {model.name}", model.name, "cpu")

    def load(self) -> None:
        info = self.probe(self.cfg, self.models_dir)
        if not info.available:
            raise BackendUnavailable(info.reason)
        # Nothing to hold in memory: the binary loads and frees per invocation.
        self._loaded = True

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
        binary = self._binary()
        model = self._model_file(self.models_dir)
        if binary is None or model is None:
            raise BackendUnavailable(self.probe(self.cfg, self.models_dir).reason)

        ai = self.cfg.visual.ai
        steps = steps or ai.steps
        width = width or ai.width
        height = height or ai.height
        threads = ai.threads or max(1, cpu_count() - 2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            binary,
            "--mode",
            "txt2img",
            "--model",
            str(model),
            "--prompt",
            prompt,
            "--negative-prompt",
            negative_prompt,
            "--steps",
            str(steps),
            "--cfg-scale",
            str(ai.guidance_scale),
            "--width",
            str(width),
            "--height",
            str(height),
            "--seed",
            str(seed),
            "--threads",
            str(threads),
            "--output",
            str(out_path),
        ]

        started = time.monotonic()
        try:
            run(
                argv,
                RunOptions(
                    timeout_s=self.cfg.tools.timeout_s,
                    nice=self.cfg.cpu.generator_nice,
                    ionice_class=self.cfg.cpu.generator_ionice_class,
                ),
            )
        except CommandError as exc:
            raise BackendUnavailable(f"stable-diffusion.cpp failed: {exc}") from exc

        if not out_path.is_file():
            raise BackendUnavailable("stable-diffusion.cpp reported success but wrote no file")

        image = GeneratedImage(
            path=out_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            backend=self.name,
            model_id=model.name,
            seconds=time.monotonic() - started,
        )
        image.write_sidecar()
        return image
