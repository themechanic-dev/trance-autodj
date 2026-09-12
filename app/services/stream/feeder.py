"""Feeds finished blocks into ffmpeg, forever.

The feeder is deliberately the dumbest component in the system: it picks a
block, copies its bytes into a pipe, updates a counter, and picks another. It
does no decoding, no muxing and no composition, because every transition the
viewer sees was baked in hours earlier by the generator.

That is the whole reason the CPU stays quiet during a broadcast.
"""

from __future__ import annotations

import contextlib
import errno
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Config
from app.core.db import session_scope
from app.core.logging import get_logger
from app.core.paths import Paths
from app.services.visual.encoder import EncodeProfile
from app.services.visual.pool import BlockPool

log = get_logger(__name__)

# Bytes copied per write. Large enough to be cheap, small enough that a stop
# request is honoured promptly.
CHUNK_BYTES = 188 * 1024  # a whole number of MPEG-TS packets

# How long to wait for a block to appear when the pool is empty.
EMPTY_POOL_WAIT_S = 5.0


@dataclass
class FeederStats:
    blocks_played: int = 0
    bytes_written: int = 0
    current_block: str = ""
    current_started_at: float = 0.0
    last_error: str = ""
    waiting_for_blocks: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "blocks_played": self.blocks_played,
            "megabytes_written": round(self.bytes_written / 1024**2, 1),
            "current_block": self.current_block,
            "current_block_elapsed_s": (
                round(time.time() - self.current_started_at, 1) if self.current_started_at else 0.0
            ),
            "waiting_for_blocks": self.waiting_for_blocks,
            "last_error": self.last_error,
        }


class Feeder:
    """Writes block bytes into a FIFO on its own thread."""

    def __init__(self, cfg: Config, paths: Paths, profile: EncodeProfile) -> None:
        self.cfg = cfg
        self.paths = paths
        self.pool = BlockPool(cfg, paths, profile)
        self.stats = FeederStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rng = random.Random()
        # Blocks already played this cycle; cleared when every block has had a
        # turn, so the pool is heard through before anything repeats.
        self._played_this_cycle: set[str] = set()

    # -- fifo --------------------------------------------------------------

    def ensure_fifo(self) -> Path:
        path = self.paths.video_fifo
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not path.is_fifo():
            path.unlink()
        if not path.exists():
            os.mkfifo(path, 0o600)
        return path

    # -- selection ---------------------------------------------------------

    def has_blocks(self) -> bool:
        """Is there anything at all to play?

        Starting a broadcast or a monitor with an empty pool produces a pipe
        nobody writes to and a viewer staring at nothing, with no clue why.
        """
        return self.next_block() is not None

    def next_block(self) -> tuple[str, Path] | None:
        """Weighted shuffle, without repeating until the cycle is exhausted."""
        with session_scope() as session:
            candidates = self.pool.pick(session, count=20, rng=self._rng)
            if not candidates:
                return None
            fresh = [b for b in candidates if b.id not in self._played_this_cycle]
            if not fresh:
                self._played_this_cycle.clear()
                fresh = candidates
            chosen = fresh[0]
            path = self.pool.path_of(chosen)
            block_id = chosen.id

        if not path.is_file():
            log.warning("block file is missing, skipping", extra={"block": block_id})
            self._played_this_cycle.add(block_id)
            return None
        return block_id, path

    # -- the loop ----------------------------------------------------------

    def _run(self) -> None:
        fifo = self.ensure_fifo()
        log.info("feeder opening the pipe", extra={"fifo": str(fifo)})

        # Opening a FIFO for writing blocks until a reader arrives. That is
        # exactly the synchronisation we want: the feeder starts the moment
        # ffmpeg is ready, and not a moment earlier.
        try:
            handle = fifo.open("wb", buffering=0)
        except OSError as exc:
            self.stats.last_error = f"could not open the pipe: {exc}"
            log.error("feeder could not open the pipe: %s", exc)
            return

        try:
            while not self._stop.is_set():
                selection = self.next_block()
                if selection is None:
                    self.stats.waiting_for_blocks = True
                    log.warning("no blocks in the pool; waiting")
                    if self._stop.wait(EMPTY_POOL_WAIT_S):
                        break
                    continue

                self.stats.waiting_for_blocks = False
                block_id, path = selection
                self.stats.current_block = block_id
                self.stats.current_started_at = time.time()
                log.info("feeding block", extra={"block": block_id})

                if not self._write_block(handle, path):
                    break

                self._played_this_cycle.add(block_id)
                self.stats.blocks_played += 1
                with session_scope() as session:
                    self.pool.mark_played(session, block_id)
        finally:
            with contextlib.suppress(OSError):
                handle.close()
            log.info("feeder stopped", extra=self.stats.as_dict())

    def _write_block(self, handle, path: Path) -> bool:
        """Copy one block into the pipe. False means "stop the feeder"."""
        try:
            with path.open("rb") as source:
                while not self._stop.is_set():
                    chunk = source.read(CHUNK_BYTES)
                    if not chunk:
                        return True
                    handle.write(chunk)
                    self.stats.bytes_written += len(chunk)
        except BrokenPipeError:
            # ffmpeg went away. The watchdog restarts it and us with it.
            self.stats.last_error = "ffmpeg closed the pipe"
            log.warning("the pipe was closed by ffmpeg; the feeder is stopping")
            return False
        except OSError as exc:
            if exc.errno == errno.EPIPE:
                self.stats.last_error = "ffmpeg closed the pipe"
                return False
            self.stats.last_error = str(exc)
            log.error("feeder write failed: %s", exc)
            return False
        return True

    # -- control -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="feeder", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Unblock a write that is waiting on a full pipe by draining it.
            self._drain_fifo()
            thread.join(timeout=timeout_s)
        self._thread = None

    def _drain_fifo(self) -> None:
        """Read and discard whatever is in the pipe so a blocked write returns."""
        try:
            fd = os.open(self.paths.video_fifo, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    if not os.read(fd, CHUNK_BYTES):
                        break
                except BlockingIOError:
                    break
                except OSError:
                    break
        finally:
            os.close(fd)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


__all__ = ["CHUNK_BYTES", "Feeder", "FeederStats"]
