"""System endpoints: health, capabilities, metrics, preflight."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import get_runtime
from app.core import hardware
from app.core.runtime import Runtime
from app.services.system import metrics as metrics_service
from app.services.system.preflight import run_preflight

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Deliberately does no work."""
    return {"status": "ok", "version": __version__}


@router.get("/capabilities")
def capabilities(runtime: Runtime = Depends(get_runtime)) -> dict[str, object]:
    cfg = runtime.config
    caps = hardware.detect(ffmpeg=cfg.tools.ffmpeg)
    encoder, encoder_reason = hardware.resolve_encoder(cfg.video.encoder, ffmpeg=cfg.tools.ffmpeg)
    backend, backend_reason = hardware.resolve_ai_backend(cfg.visual.ai.backend)
    return {
        "cpu": {
            "count": caps.cpu.count,
            "model": caps.cpu.model,
            "architecture": caps.cpu.architecture,
        },
        "gpu": {
            "present": caps.gpu.present,
            "name": caps.gpu.name,
            "driver": caps.gpu.driver,
            "memory_mb": caps.gpu.memory_mb,
        },
        "ffmpeg": {"path": caps.ffmpeg_path, "version": caps.ffmpeg_version},
        "encoder": {"resolved": encoder, "reason": encoder_reason},
        "ai_backend": {"resolved": backend, "reason": backend_reason},
        "nvenc": {"available": caps.nvenc_available, "reason": caps.nvenc_reason},
        "video_profile": {
            "resolution": cfg.video.resolution,
            "fps": cfg.video.fps,
            "bitrate_k": cfg.video.bitrate_k,
        },
    }


@router.get("/metrics")
def system_metrics(runtime: Runtime = Depends(get_runtime)) -> dict[str, object]:
    return metrics_service.collect(runtime.paths.root).as_dict()


@router.get("/preflight")
def preflight(runtime: Runtime = Depends(get_runtime)) -> dict[str, object]:
    return run_preflight(runtime.config, runtime.paths).as_dict()
