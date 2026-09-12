"""Supervised child processes for Icecast and Liquidsoap.

The application owns these: it writes their configuration, starts them, reads
their output into our own structured log, and restarts them when they die.
That is why everything lives in one container — the dashboard has to be able
to stop and start the audio stack, and handing a web process the Docker socket
to do that would be far worse.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger, redact
from app.core.proc import require

log = get_logger(__name__)

# stderr/stdout lines kept per process for the dashboard's log view.
LOG_TAIL = 300

# Restart policy for a child that dies on its own.
RESTART_MIN_S = 2.0
RESTART_MAX_S = 60.0

# A child that stays up this long has "succeeded"; reset its backoff.
HEALTHY_AFTER_S = 30.0


@dataclass
class ProcessStatus:
    name: str
    running: bool
    pid: int | None
    started_at: float | None
    restarts: int
    last_error: str
    uptime_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "running": self.running,
            "pid": self.pid,
            "restarts": self.restarts,
            "last_error": self.last_error,
            "uptime_s": round(self.uptime_s, 1),
        }


@dataclass
class SupervisedProcess:
    """One child process, restarted with exponential backoff when it dies."""

    name: str
    argv: Sequence[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None
    # Sent first on stop. Liquidsoap and Icecast both shut down cleanly on it.
    stop_signal: int = signal.SIGTERM
    stop_grace_s: float = 15.0
    autorestart: bool = True

    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _reader: threading.Thread | None = field(default=None, init=False, repr=False)
    _supervisor: threading.Thread | None = field(default=None, init=False, repr=False)
    _stopping: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lines: deque[str] = field(
        default_factory=lambda: deque(maxlen=LOG_TAIL), init=False, repr=False
    )
    _started_at: float | None = field(default=None, init=False, repr=False)
    _restarts: int = field(default=0, init=False, repr=False)
    _last_error: str = field(default="", init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        require(self.argv[0])
        self._stopping.clear()
        self._spawn()
        if self.autorestart and (self._supervisor is None or not self._supervisor.is_alive()):
            self._supervisor = threading.Thread(
                target=self._supervise, name=f"{self.name}-supervisor", daemon=True
            )
            self._supervisor.start()

    def _spawn(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            environment = dict(os.environ)
            if self.env:
                environment.update(self.env)
            log.info("starting %s", self.name, extra={"argv": redact(" ".join(self.argv))})
            self._process = subprocess.Popen(  # noqa: S603 - argv is a list
                list(self.argv),
                cwd=str(self.cwd) if self.cwd else None,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self._started_at = time.time()
        self._reader = threading.Thread(target=self._drain, name=f"{self.name}-log", daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw in process.stdout:
            line = redact(raw.decode("utf-8", errors="replace").rstrip())
            if not line:
                continue
            self._lines.append(line)
            # Their own severity is in the text; ours is uniformly "the child
            # said something", so keep it at debug and let the UI show the tail.
            log.debug("%s: %s", self.name, line)
            lowered = line.lower()
            if "error" in lowered or "fatal" in lowered:
                self._last_error = line

    def _supervise(self) -> None:
        delay = RESTART_MIN_S
        while not self._stopping.is_set():
            process = self._process
            if process is None:
                break
            code = process.wait()
            if self._stopping.is_set():
                break

            alive_for = time.time() - (self._started_at or time.time())
            if alive_for >= HEALTHY_AFTER_S:
                delay = RESTART_MIN_S  # it was healthy; treat this as a fresh fault

            self._restarts += 1
            log.warning(
                "%s exited unexpectedly, restarting",
                self.name,
                extra={
                    "exit_code": code,
                    "ran_for_s": round(alive_for, 1),
                    "restart_in_s": delay,
                    "restarts": self._restarts,
                },
            )
            if self._stopping.wait(delay):
                break
            delay = min(delay * 2, RESTART_MAX_S)
            self._spawn()

    def stop(self) -> None:
        self._stopping.set()
        self._terminate(self._process)
        self._join_supervisor()
        # The supervisor may already have been inside _spawn() when the flag
        # was set, in which case it started one last child before noticing.
        self._terminate(self._process)

    def _terminate(self, process: subprocess.Popen[bytes] | None) -> None:
        if process is None or process.poll() is not None:
            return
        log.info("stopping %s", self.name)
        with contextlib.suppress(ProcessLookupError, OSError):
            process.send_signal(self.stop_signal)
        try:
            process.wait(timeout=self.stop_grace_s)
        except subprocess.TimeoutExpired:
            log.warning("%s ignored %s, killing", self.name, self.stop_signal)
            with contextlib.suppress(ProcessLookupError, OSError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)

    def _join_supervisor(self) -> None:
        """Wait for the restart loop to notice that it has been stopped.

        Without this, restart() races it. stop() sets the flag and start()
        clears it again a moment later, so the old loop can wake up to a clear
        flag and carry on: two supervisors then restart two children that
        fight over the same port forever, and neither can be stopped because
        only one of them is still reachable. The mirror image is just as bad —
        the old thread is still alive when start() checks, so start() skips
        creating a supervisor and the child is left unsupervised.
        """
        supervisor = self._supervisor
        if supervisor is None or supervisor is threading.current_thread():
            return
        supervisor.join(timeout=self.stop_grace_s + 5.0)
        if supervisor.is_alive():
            # Leave it recorded: start() will then decline to add a second
            # loop, which is the safer of the two ways to be wrong.
            log.warning("%s supervisor did not stop", self.name)
        else:
            self._supervisor = None

    def restart(self) -> None:
        self.stop()
        self.start()

    # -- introspection -----------------------------------------------------

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def uptime_s(self) -> float:
        return (time.time() - self._started_at) if (self.running and self._started_at) else 0.0

    def status(self) -> ProcessStatus:
        # A line containing the word "error" during startup is usually a
        # notice, not a fault. Once the process has been up and healthy for a
        # while, stop presenting it as the current state — otherwise a benign
        # first-second message sits in the dashboard for months.
        settled = self.running and self.uptime_s >= HEALTHY_AFTER_S
        return ProcessStatus(
            name=self.name,
            running=self.running,
            pid=self._process.pid if self._process else None,
            started_at=self._started_at,
            restarts=self._restarts,
            last_error="" if settled else self._last_error,
            uptime_s=self.uptime_s,
        )

    def tail(self, count: int = 50) -> list[str]:
        return list(self._lines)[-count:]


__all__ = ["RESTART_MAX_S", "RESTART_MIN_S", "ProcessStatus", "SupervisedProcess"]
