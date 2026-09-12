"""Live log tail, over server-sent events.

SSE rather than a websocket: it is one-directional, it reconnects on its own,
and it is a plain GET — which means it works through any proxy without special
configuration. For following a log that is all it needs to be.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_runtime
from app.core.logging import get_logger
from app.core.runtime import Runtime

log = get_logger(__name__)

router = APIRouter(prefix="/api/logs", tags=["logs"])

POLL_S = 1.0
MAX_TAIL_LINES = 2000
# Sent when nothing has happened, so proxies do not time the connection out.
HEARTBEAT_EVERY_S = 20.0


def _log_files(runtime: Runtime) -> dict[str, Path]:
    return {path.stem: path for path in sorted(runtime.paths.logs.glob("*.log"))}


@router.get("/services")
def services(runtime: Annotated[Runtime, Depends(get_runtime)]) -> dict[str, Any]:
    return {
        "services": [
            {"name": name, "size_kb": round(path.stat().st_size / 1024, 1)}
            for name, path in _log_files(runtime).items()
        ]
    }


def _tail(path: Path, lines: int) -> list[str]:
    """Read the last N lines without loading the whole file."""
    if not path.is_file():
        return []
    chunk = 64 * 1024
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        data = b""
        while size > 0 and data.count(b"\n") <= lines:
            step = min(chunk, size)
            size -= step
            handle.seek(size)
            data = handle.read(step) + data
    return data.decode("utf-8", errors="replace").splitlines()[-lines:]


@router.get("/{service}")
def read_log(
    service: str,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    lines: int = Query(default=200, ge=1, le=MAX_TAIL_LINES),
    level: str | None = None,
) -> dict[str, Any]:
    files = _log_files(runtime)
    if service not in files:
        raise HTTPException(404, f"no log for {service!r}")
    raw = _tail(files[service], lines)
    entries = [_parse(line) for line in raw]
    if level:
        wanted = {"debug": 0, "info": 1, "warning": 2, "error": 3}
        threshold = wanted.get(level.lower(), 0)
        entries = [e for e in entries if wanted.get(e["level"], 1) >= threshold]
    return {"service": service, "entries": entries}


def _parse(line: str) -> dict[str, Any]:
    """Our own logs are JSON; anything else is shown verbatim."""
    try:
        data = json.loads(line)
    except ValueError:
        return {"ts": "", "level": "info", "logger": "", "msg": line, "extra": {}}
    if not isinstance(data, dict):
        return {"ts": "", "level": "info", "logger": "", "msg": line, "extra": {}}
    known = {"ts", "level", "logger", "service", "msg", "exc"}
    return {
        "ts": data.get("ts", ""),
        "level": data.get("level", "info"),
        "logger": data.get("logger", ""),
        "msg": data.get("msg", ""),
        "exc": data.get("exc", ""),
        "extra": {k: v for k, v in data.items() if k not in known},
    }


@router.get("/{service}/stream")
async def stream_log(
    service: str,
    request: Request,
    runtime: Annotated[Runtime, Depends(get_runtime)],
) -> StreamingResponse:
    files = _log_files(runtime)
    if service not in files:
        raise HTTPException(404, f"no log for {service!r}")
    path = files[service]

    async def events() -> AsyncIterator[str]:
        position = path.stat().st_size if path.is_file() else 0
        idle = 0.0
        while True:
            if await request.is_disconnected():
                break
            try:
                size = path.stat().st_size
                if size < position:
                    # The file was rotated; start again from the beginning.
                    position = 0
                if size > position:
                    with path.open("rb") as handle:
                        handle.seek(position)
                        chunk = handle.read(size - position)
                        position = size
                    for line in chunk.decode("utf-8", errors="replace").splitlines():
                        if line.strip():
                            yield f"data: {json.dumps(_parse(line))}\n\n"
                    idle = 0.0
            except OSError:
                pass

            await asyncio.sleep(POLL_S)
            idle += POLL_S
            if idle >= HEARTBEAT_EVERY_S:
                idle = 0.0
                yield ": heartbeat\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
