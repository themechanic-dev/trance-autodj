"""Visual pool endpoints: what exists, what it looks like, and make more."""

from __future__ import annotations

import json
import subprocess
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select

from app.api.deps import get_runtime
from app.core.db import get_session, session_scope
from app.core.logging import get_logger
from app.core.runtime import Runtime
from app.models.entities import Block, BlockStatus, GenerationJob, JobKind, JobStatus, utcnow
from app.services.visual import ai as ai_backends
from app.services.visual import ai_source, procedural
from app.services.visual.ai.prompts import PromptSet
from app.services.visual.encoder import EncodeProfile
from app.services.visual.palettes import PaletteSet
from app.services.visual.pool import BlockPool
from app.services.visual.worker import build_one_block

log = get_logger(__name__)

router = APIRouter(prefix="/api/visuals", tags=["visuals"])

MAX_MANUAL_BLOCKS = 20


def _pool(runtime: Runtime) -> BlockPool:
    profile = EncodeProfile.from_config(runtime.config)
    return BlockPool(runtime.config, runtime.paths, profile)


def _block_json(block: Block, runtime: Runtime) -> dict[str, Any]:
    try:
        metadata = json.loads(block.metadata_json)
    except ValueError:
        metadata = {}
    return {
        "id": block.id,
        "duration_s": round(block.duration_s, 1),
        "size_mb": round(block.size_bytes / 1024**2, 1),
        "source": block.source.value,
        "status": block.status.value,
        "reject_reason": block.reject_reason,
        "play_count": block.play_count,
        "pinned": block.pinned,
        "created_at": block.created_at.isoformat(),
        "last_played_at": block.last_played_at.isoformat() if block.last_played_at else None,
        "has_thumbnail": bool(block.thumbnail_relpath)
        and (runtime.paths.root / block.thumbnail_relpath).is_file(),
        "generators": sorted(
            {c.get("generator", "") for c in metadata.get("clips", []) if c.get("generator")}
        ),
        "palettes": sorted(
            {c.get("palette", "") for c in metadata.get("clips", []) if c.get("palette")}
        ),
        "transitions": metadata.get("transitions", []),
        "clips": len(metadata.get("clips", [])),
    }


@router.get("/pool")
def pool_stats(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    with session_scope() as session:
        pool = _pool(runtime)
        pool.reconcile(session)
        return pool.stats(session).as_dict()


@router.get("/blocks")
def list_blocks(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
    status: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    statement = select(Block).order_by(Block.created_at.desc()).limit(limit)
    if status:
        try:
            statement = statement.where(Block.status == BlockStatus(status))
        except ValueError:
            raise HTTPException(400, f"unknown status {status!r}") from None
    blocks = session.exec(statement).all()
    return {"blocks": [_block_json(b, runtime) for b in blocks], "count": len(blocks)}


@router.get("/generators")
def list_generators() -> dict[str, Any]:
    return {
        "generators": [
            {
                "name": name,
                "description": cls.description,
                "loops_natively": cls.loops_natively,
            }
            for name, cls in sorted(procedural.available().items())
        ]
    }


@router.get("/palettes")
def list_palettes(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    from pathlib import Path

    palettes = PaletteSet.load(Path(runtime.config.app.config_dir) / "palettes.json")
    return {
        "palettes": [
            {"id": p.id, "name": p.name, "colors": list(p.colors), "weight": p.weight}
            for p in palettes
        ]
    }


@router.get("/jobs")
def list_jobs(
    session: Annotated[Session, Depends(get_session)],
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    jobs = session.exec(
        select(GenerationJob).order_by(GenerationJob.created_at.desc()).limit(limit)
    ).all()
    return {
        "jobs": [
            {
                "id": j.id,
                "kind": j.kind.value,
                "status": j.status.value,
                "progress": round(j.progress, 3),
                "step": j.step,
                "result_id": j.result_id,
                "error": j.error,
                "created_at": j.created_at.isoformat(),
                "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            }
            for j in jobs
        ],
        "running": sum(1 for j in jobs if j.status is JobStatus.RUNNING),
    }


@router.get("/blocks/{block_id}/thumbnail")
def block_thumbnail(
    block_id: str,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> FileResponse:
    block = session.get(Block, block_id)
    if block is None or not block.thumbnail_relpath:
        raise HTTPException(404, "no thumbnail for that block")
    path = runtime.paths.root / block.thumbnail_relpath
    if not path.is_file():
        raise HTTPException(404, "the thumbnail file is gone")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/blocks/{block_id}/preview.mp4")
def block_preview(
    block_id: str,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> StreamingResponse:
    """Remux the block to fragmented MP4 so a browser can play it.

    Browsers cannot play MPEG-TS, but the video inside is ordinary H.264, so
    this is a container change with ``-c copy`` — no re-encode, and therefore
    no meaningful CPU cost even while streaming.
    """
    block = session.get(Block, block_id)
    if block is None:
        raise HTTPException(404, "no such block")
    path = runtime.paths.root / block.relpath
    if not path.is_file():
        raise HTTPException(404, "the block file is gone")

    argv = [
        runtime.config.tools.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-fflags",
        "+genpts",
        "-i",
        str(path),
        "-c",
        "copy",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
    ]

    def stream():
        process = subprocess.Popen(  # noqa: S603 - argv is a list, never a shell string
            argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read(64 * 1024):
                yield chunk
        finally:
            # The client can close the tab mid-playback; do not leave ffmpeg
            # running for a viewer that has gone away.
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)

    return StreamingResponse(stream(), media_type="video/mp4")


@router.post("/blocks/{block_id}/pin")
def pin_block(
    block_id: str,
    session: Annotated[Session, Depends(get_session)],
    pinned: bool = True,
) -> dict[str, Any]:
    block = session.get(Block, block_id)
    if block is None:
        raise HTTPException(404, "no such block")
    block.pinned = pinned
    session.add(block)
    return {"id": block_id, "pinned": pinned}


@router.delete("/blocks/{block_id}")
def delete_block(
    block_id: str,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    block = session.get(Block, block_id)
    if block is None:
        raise HTTPException(404, "no such block")
    _pool(runtime)._delete_files(block)
    session.delete(block)
    return {"deleted": block_id}


@router.post("/generate")
def generate_blocks(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    background: BackgroundTasks,
    count: int = Query(default=1, ge=1, le=MAX_MANUAL_BLOCKS),
) -> dict[str, Any]:
    """Build blocks now, ignoring the load brake — the operator asked.

    Runs in the background so the request returns immediately; watch /jobs
    for progress.
    """
    background.add_task(_generate_in_background, runtime, count)
    return {"queued": count, "note": "watch /api/visuals/jobs for progress"}


def _generate_in_background(runtime: Runtime, count: int) -> None:
    from pathlib import Path

    profile = EncodeProfile.from_config(runtime.config)
    palettes = PaletteSet.load(Path(runtime.config.app.config_dir) / "palettes.json")
    pool = BlockPool(runtime.config, runtime.paths, profile)
    ai = ai_source.build(runtime.config, runtime.paths)

    for index in range(count):
        # A row per block, exactly as the worker does, so the dashboard shows
        # progress for a hand-triggered build too instead of a silent wait.
        with session_scope() as session:
            job = GenerationJob(kind=JobKind.BLOCK, status=JobStatus.RUNNING, started_at=utcnow())
            session.add(job)
            session.flush()
            job_id = job.id

        try:
            block_id = build_one_block(
                runtime,
                profile=profile,
                palettes=palettes,
                pool=pool,
                job_id=job_id,
                ai=ai,
            )
            status, error = (JobStatus.DONE if block_id else JobStatus.CANCELLED), ""
            log.info("manual block %d/%d finished", index + 1, count, extra={"block": block_id})
        except Exception as exc:
            block_id, status, error = None, JobStatus.FAILED, str(exc)[:2000]
            log.exception("manual block %d/%d failed", index + 1, count)

        with session_scope() as session:
            job = session.get(GenerationJob, job_id)
            if job is not None:
                job.status = status
                job.progress = 1.0
                job.step = "done" if status is JobStatus.DONE else status.value
                job.result_id = block_id or ""
                job.error = error
                job.finished_at = utcnow()
                session.add(job)


@router.get("/ai")
def ai_status(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    """What every AI backend reports, and which one would be chosen.

    "none" is a normal answer, not a failure: procedural visuals need no model.
    """
    from pathlib import Path

    cfg = runtime.config
    backend, reason = ai_backends.resolve(cfg, runtime.paths.models)
    prompts = PromptSet.load(Path(cfg.app.config_dir) / "prompts.json")
    return {
        "resolved": backend.name if backend else "none",
        "reason": reason,
        "configured": cfg.visual.ai.backend,
        "ai_ratio": cfg.visual.sources.ai_ratio,
        "model_id": cfg.visual.ai.model_id,
        "models_dir": str(runtime.paths.models),
        "backends": [info.as_dict() for info in ai_backends.probe_all(cfg, runtime.paths.models)],
        "presets": [p.as_dict() for p in prompts],
        "negative_prompt": prompts.negative_prompt,
    }
