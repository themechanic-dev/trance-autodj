"""Reading the now-playing state Liquidsoap writes.

Liquidsoap writes the file atomically on every track change; this side only
reads. Keeping the exchange to a file means the dashboard works even while
Liquidsoap is restarting, and there is no protocol to keep in sync.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class NowPlaying:
    artist: str = ""
    title: str = ""
    album: str = ""
    filename: str = ""
    bpm: str = ""
    started_at: float = 0.0
    source: str = ""

    @property
    def display(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or self.filename or "—"

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.time() - self.started_at) if self.started_at else 0.0

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["display"] = self.display
        data["elapsed_s"] = round(self.elapsed_s, 1)
        return data


def read(path: Path) -> NowPlaying:
    """Never raises: an unreadable file means "nothing is playing yet"."""
    if not path.is_file():
        return NowPlaying()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Torn or absent. Liquidsoap writes atomically, so this is rare, and
        # the caller polls anyway.
        return NowPlaying()
    if not isinstance(data, dict):
        return NowPlaying()
    return NowPlaying(
        artist=str(data.get("artist", "")),
        title=str(data.get("title", "")),
        album=str(data.get("album", "")),
        filename=str(data.get("filename", "")),
        bpm=str(data.get("bpm", "")),
        started_at=float(data.get("started_at", 0.0) or 0.0),
        source=str(data.get("source", "")),
    )


def write(path: Path, now_playing: NowPlaying) -> None:
    """Used by tests and by the fallback path, not by Liquidsoap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(now_playing), ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


__all__ = ["NowPlaying", "read", "write"]
