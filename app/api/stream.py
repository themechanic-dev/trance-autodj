"""Broadcast endpoints: go live, stop, health, and the stream key."""

from __future__ import annotations

import time
from typing import Annotated, Any

from anyio.to_thread import run_sync
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_runtime
from app.core.logging import get_logger
from app.core.runtime import (
    SECRET_STREAM_KEY,
    SECRET_YT_CLIENT_ID,
    SECRET_YT_CLIENT_SECRET,
    SECRET_YT_REFRESH_TOKEN,
    Runtime,
)
from app.core.security import mask
from app.services.stream import youtube
from app.services.stream.manager import StreamManager
from app.services.stream.streamer import build_command
from app.services.visual.encoder import EncodeProfile

log = get_logger(__name__)

router = APIRouter(prefix="/api/stream", tags=["stream"])

MIN_STREAM_KEY_LENGTH = 8

# A preview is something you watch for a minute, not a second broadcast.
# The cap is a backstop for a tab left open on a forgotten machine.
MONITOR_MAX_S = 30 * 60


def get_stream(request: Request) -> StreamManager:
    manager = getattr(request.app.state, "stream", None)
    if manager is None:
        raise HTTPException(503, "the streamer is not available in this process")
    return manager


@router.get("/status")
def status(stream: Annotated[StreamManager, Depends(get_stream)]) -> dict[str, Any]:
    return stream.status()


@router.post("/start")
def start(stream: Annotated[StreamManager, Depends(get_stream)]) -> dict[str, Any]:
    started, reason = stream.start()
    if not started:
        raise HTTPException(409, reason)
    return {"started": True, "reason": reason, **stream.status()}


@router.post("/stop")
def stop(stream: Annotated[StreamManager, Depends(get_stream)]) -> dict[str, Any]:
    stream.stop()
    return stream.status()


@router.get("/logs")
def logs(
    stream: Annotated[StreamManager, Depends(get_stream)],
    count: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    return {"lines": stream.logs(count)}


@router.get("/command")
def command(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    """The exact ffmpeg command, with the key redacted.

    Useful when something is wrong and you want to run it by hand.
    """
    profile = EncodeProfile.from_config(runtime.config)
    argv = build_command(runtime.config, runtime.paths, profile, stream_key="STREAM-KEY-REDACTED")
    return {"command": argv, "video_mode": runtime.config.stream.video_mode}


@router.post("/monitor/keepalive")
def monitor_keepalive(request: Request) -> dict[str, Any]:
    """Yes, somebody is still watching."""
    request.app.state.stream.monitor_heartbeat()
    return {"ok": True}


@router.post("/monitor/stop")
def monitor_stop(request: Request) -> dict[str, Any]:
    """Stop a monitor whose viewer is gone, or that was simply forgotten."""
    stream: StreamManager = request.app.state.stream
    return {"stopped": stream.stop_monitor()}


class YoutubeCredentials(BaseModel):
    """The operator's own Google Cloud client pair, in the body, never a URL."""

    client_id: str = Field(..., min_length=10)
    client_secret: str = Field(..., min_length=10)


@router.get("/youtube")
def youtube_status(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    secrets = runtime.secrets
    youtube_cfg = runtime.config.stream.youtube
    connected = bool(
        secrets.get(SECRET_YT_CLIENT_ID)
        and secrets.get(SECRET_YT_CLIENT_SECRET)
        and secrets.get(SECRET_YT_REFRESH_TOKEN)
    )
    return {
        "has_client": bool(secrets.get(SECRET_YT_CLIENT_ID)),
        "connected": connected,
        "title_template": youtube_cfg.title_template,
        "description_template": youtube_cfg.description_template,
        # What the title would be if the broadcast started now, so the
        # operator can see the stamp rather than guess at the placeholders.
        "preview": youtube.render_title(
            youtube_cfg.title_template, timezone=runtime.config.app.timezone
        ),
    }


@router.put("/youtube/client")
def set_youtube_client(
    payload: YoutubeCredentials,
    runtime: Annotated[Runtime, Depends(get_runtime)],
) -> dict[str, Any]:
    runtime.secrets.set(SECRET_YT_CLIENT_ID, payload.client_id.strip())
    runtime.secrets.set(SECRET_YT_CLIENT_SECRET, payload.client_secret.strip())
    # Changing the client invalidates any approval given to the old one.
    runtime.secrets.delete(SECRET_YT_REFRESH_TOKEN)
    log.info("YouTube client credentials stored")  # never the values
    return {"has_client": True, "connected": False}


@router.post("/youtube/connect")
def youtube_connect(request: Request) -> dict[str, Any]:
    """Start the device flow and hand back the code to type on a phone."""
    runtime: Runtime = request.app.state.runtime
    client_id = runtime.secrets.get(SECRET_YT_CLIENT_ID)
    if not client_id:
        raise HTTPException(409, "add the Google client ID and secret first")
    try:
        code = youtube.begin_device_flow(client_id)
    except youtube.YoutubeError as exc:
        raise HTTPException(502, str(exc)) from exc
    # Held in memory only: it is short-lived and worth nothing once used.
    request.app.state.youtube_device = code
    return code.as_dict()


@router.post("/youtube/poll")
def youtube_poll(request: Request) -> dict[str, Any]:
    """Has the operator approved it yet?"""
    runtime: Runtime = request.app.state.runtime
    code = getattr(request.app.state, "youtube_device", None)
    if code is None:
        raise HTTPException(409, "no connection is in progress")
    try:
        token = youtube.poll_device_flow(
            runtime.secrets.get(SECRET_YT_CLIENT_ID),
            runtime.secrets.get(SECRET_YT_CLIENT_SECRET),
            code.device_code,
        )
    except youtube.YoutubeError as exc:
        request.app.state.youtube_device = None
        raise HTTPException(502, str(exc)) from exc
    if token is None:
        return {"connected": False, "waiting": True}
    runtime.secrets.set(SECRET_YT_REFRESH_TOKEN, token)
    request.app.state.youtube_device = None
    log.info("YouTube account connected")
    return {"connected": True, "waiting": False}


@router.delete("/youtube")
def youtube_disconnect(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    for name in (SECRET_YT_REFRESH_TOKEN, SECRET_YT_CLIENT_SECRET, SECRET_YT_CLIENT_ID):
        runtime.secrets.delete(name)
    return {"has_client": False, "connected": False}


@router.get("/monitor/status")
def monitor_status(request: Request) -> dict[str, Any]:
    stream: StreamManager = request.app.state.stream
    ready, reason = stream.monitor_available()
    audio = request.app.state.audio.status()
    return {
        "ready": ready and audio.on_air,
        "reason": reason if not ready else ("" if audio.on_air else "the audio is not on air"),
        "url": "/api/stream/monitor/live.mp4",
    }


@router.get("/monitor/live.mp4")
def monitor_live(request: Request) -> StreamingResponse:
    """What would be sent to YouTube, played here instead.

    The pipeline runs for exactly as long as this response is open: closing
    the tab stops ffmpeg and the feeder. There is nothing to remember to turn
    off, and nothing keeps burning CPU because a preview was left in a
    background tab.
    """
    stream: StreamManager = request.app.state.stream

    async def chunks():
        try:
            with stream.monitor() as process:
                assert process.stdout is not None
                started = time.monotonic()
                while True:
                    # A browser that closes the tab does not always tell us,
                    # so ask. Without this the kernel quietly buffers writes
                    # nobody reads and ffmpeg runs on for minutes.
                    if await request.is_disconnected():
                        log.info("monitor viewer went away")
                        break
                    if time.monotonic() - started > MONITOR_MAX_S:
                        log.info("monitor reached its time limit")
                        break
                    data = await run_sync(process.stdout.read, 64 * 1024)
                    if not data:
                        break
                    yield data
        except RuntimeError as exc:
            log.info("monitor refused: %s", exc)
            return

    ready, reason = stream.monitor_available()
    if not ready:
        raise HTTPException(409, reason)
    return StreamingResponse(
        chunks(),
        media_type="video/mp4",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/key")
def key_status(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    """Whether a key is set, and its last few characters. Never the key."""
    key = runtime.stream_key()
    return {
        "configured": bool(key),
        "masked": mask(key),
        "from_environment": bool(runtime.config.stream.youtube.stream_key),
        "rtmp_url": runtime.config.stream.youtube.rtmp_url,
    }


class StreamKeyIn(BaseModel):
    """The key arrives in the body, never in the query string.

    A stream key is a credential: anyone holding it can broadcast to the
    channel. Query strings are written to access logs by uvicorn and by every
    reverse proxy in front of it, and they stay in browser history — which
    would have quietly undone the encrypted store, the masking and the log
    redactor that the rest of this file goes to some trouble to provide.
    """

    key: str = Field(..., min_length=MIN_STREAM_KEY_LENGTH)


@router.put("/key")
def set_key(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    payload: StreamKeyIn,
) -> dict[str, Any]:
    key = payload.key
    if runtime.config.stream.youtube.stream_key:
        raise HTTPException(
            409,
            "the stream key is pinned by TAD_STREAM__YOUTUBE__STREAM_KEY; "
            "unset it to manage the key from here",
        )
    runtime.secrets.set(SECRET_STREAM_KEY, key.strip())
    log.info("the stream key was updated")  # never log the value
    return {"configured": True, "masked": mask(key.strip())}


@router.delete("/key")
def clear_key(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    stream: Annotated[StreamManager, Depends(get_stream)],
) -> dict[str, Any]:
    """Forget the key — and come off air, because there is nowhere to go.

    Removing the key used to leave the broadcast wanting to be live with a
    dead credential the manager still held in memory, so the watchdog kept
    relaunching ffmpeg against an ingest that rejected it instantly, once a
    minute, for as long as the container ran.
    """
    runtime.secrets.delete(SECRET_STREAM_KEY)
    was_live = stream.status().get("want_live", False)
    if was_live:
        log.info("the stream key was removed while live; going off air")
        stream.stop()
    return {"configured": False, "stopped": bool(was_live)}
