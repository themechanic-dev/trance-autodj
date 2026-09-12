"""Settings endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlmodel import Session

from app.api.deps import get_runtime
from app.core.db import get_session
from app.core.logging import get_logger
from app.core.runtime import Runtime
from app.services.audio import playlists
from app.services.system import settings as settings_service

log = get_logger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _why(exc: ValidationError) -> str:
    """The one line of a pydantic error worth showing an operator."""
    first = exc.errors()[0]
    where = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "")).removeprefix("Value error, ")
    return f"{where}: {message}" if where else message


@router.get("")
def list_settings(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    rows = settings_service.describe(session, runtime.config)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    return {"groups": groups}


@router.put("")
def update_settings(
    request: Request,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
    values: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    accepted, rejected = settings_service.set_many(session, values)
    if not accepted and rejected:
        raise HTTPException(400, "; ".join(rejected))

    # Same order as the reset below: build the configuration first, commit
    # only once it is known to be valid.
    session.flush()
    try:
        config = settings_service.apply(session)
    except ValidationError as exc:
        session.rollback()
        raise HTTPException(409, _why(exc)) from exc
    session.commit()

    # The process configuration takes effect without a restart, and Liquidsoap
    # is bounced if what changed lives in its script.
    runtime.config = config
    restarted = False
    audio = getattr(request.app.state, "audio", None)
    if audio is not None and settings_service.needs_audio_restart(accepted):
        audio.cfg = runtime.config
        audio.apply_settings()
        restarted = True

    # The generated script is not the only thing an audio setting changes: the
    # playlist file carries the annotate: values, and the switch between one
    # list and the rotation decides which playlists are in it at all. Changing
    # to "rotation" used to restart Liquidsoap with the right script and leave
    # it reading a file that still held the single list — the switch appeared
    # to do nothing.
    if any(path.startswith("audio.") for path in accepted):
        playlists.refresh_active(session, runtime.paths, runtime.config)

    needs_stream_restart = _refresh_stream(request, runtime, accepted)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "audio_restarted": restarted,
        "stream_restart_needed": needs_stream_restart,
    }


def _refresh_stream(request: Request, runtime: Runtime, paths: list[str]) -> bool:
    """Hand the new configuration to the broadcast, if it owns any of it."""
    stream = getattr(request.app.state, "stream", None)
    if stream is None or not any(p.startswith(("stream.", "video.")) for p in paths):
        return False
    return bool(stream.apply_settings(runtime.config))


@router.delete("/{path:path}")
def reset_setting(
    path: str,
    request: Request,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    if not settings_service.reset(session, path):
        raise HTTPException(404, "that setting has no override")

    # Validate before committing, the way set_many already does. *Removing* an
    # override can make the configuration invalid just as easily as adding
    # one: resetting clip.min_duration_s to its default while
    # clip.max_duration_s is still overridden below it is enough. Committing
    # first meant the caller was told 500 about a change that had in fact
    # been made.
    session.flush()
    try:
        config = settings_service.apply(session)
    except ValidationError as exc:
        session.rollback()
        raise HTTPException(409, _why(exc)) from exc

    session.commit()
    runtime.config = config
    return {"reset": path, "stream_restart_needed": _refresh_stream(request, runtime, [path])}
