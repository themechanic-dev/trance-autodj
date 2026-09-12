"""One wrapper for every external command (ffmpeg, ffprobe, liquidsoap, ...).

Nothing in this project calls subprocess directly. Everything goes through
here so that timeouts, retries, output capture and secret redaction happen
once and consistently, and so a failure produces a message that says what was
run and what came back instead of a bare CalledProcessError.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger, redact

log = get_logger(__name__)


class CommandError(RuntimeError):
    """A command exited non-zero, timed out, or was not found."""

    def __init__(
        self,
        message: str,
        *,
        argv: Sequence[str],
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.argv = list(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        parts = [super().__str__()]
        parts.append(f"command: {redact(shlex.join(self.argv))}")
        if self.returncode is not None:
            parts.append(f"exit code: {self.returncode}")
        tail = (self.stderr or self.stdout).strip()
        if tail:
            lines = tail.splitlines()[-12:]
            parts.append("output:\n  " + "\n  ".join(redact(line) for line in lines))
        return "\n".join(parts)


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def json(self) -> Any:
        return json.loads(self.stdout)


@dataclass
class RunOptions:
    timeout_s: float = 900.0
    retries: int = 0
    retry_delay_s: float = 3.0
    check: bool = True
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    nice: int | None = None
    ionice_class: str | None = None
    input_text: str | None = None
    # stderr lines kept in memory for the error message (ffmpeg is verbose).
    max_captured_lines: int = 400
    extra_log_fields: dict[str, Any] = field(default_factory=dict)


_IONICE_CLASSES = {"none": 0, "realtime": 1, "best-effort": 2, "idle": 3}


def which(binary: str) -> str | None:
    return shutil.which(binary)


def require(binary: str) -> str:
    found = which(binary)
    if not found:
        raise CommandError(f"{binary!r} was not found on PATH", argv=[binary], returncode=None)
    return found


def _wrap_priority(argv: Sequence[str], opts: RunOptions) -> list[str]:
    """Prefix with nice/ionice when asked and when those tools exist.

    Both are optional: a minimal container may not ship util-linux, and a
    missing scheduler tweak must never stop the work from happening.
    """
    prefix: list[str] = []
    if opts.ionice_class and opts.ionice_class != "none" and which("ionice"):
        cls = _IONICE_CLASSES.get(opts.ionice_class)
        if cls is not None:
            prefix += ["ionice", "-c", str(cls)]
    if opts.nice is not None and opts.nice != 0 and which("nice"):
        prefix += ["nice", "-n", str(opts.nice)]
    return prefix + list(argv)


def _build_env(opts: RunOptions) -> dict[str, str] | None:
    if opts.env is None:
        return None
    env = dict(os.environ)
    env.update(opts.env)
    return env


def run(argv: Sequence[str], opts: RunOptions | None = None) -> CommandResult:
    """Run a command to completion, with retries and a hard timeout."""
    opts = opts or RunOptions()
    argv = list(argv)
    require(argv[0])
    full = _wrap_priority(argv, opts)

    last_error: CommandError | None = None
    for attempt in range(1, opts.retries + 2):
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - argv is a list, never a shell string
                full,
                check=False,  # the return code is inspected below
                capture_output=True,
                text=True,
                errors="replace",
                timeout=opts.timeout_s,
                cwd=str(opts.cwd) if opts.cwd else None,
                env=_build_env(opts),
                input=opts.input_text,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            last_error = CommandError(
                f"timed out after {opts.timeout_s:.0f}s",
                argv=full,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
            )
        else:
            duration = time.monotonic() - started
            result = CommandResult(
                argv=full,
                returncode=completed.returncode,
                stdout=completed.stdout or "",
                stderr=_tail(completed.stderr or "", opts.max_captured_lines),
                duration_s=duration,
                attempts=attempt,
            )
            if result.ok or not opts.check:
                log.debug(
                    "command finished",
                    extra={
                        "argv": redact(shlex.join(full)),
                        "rc": result.returncode,
                        "duration_s": round(duration, 3),
                        **opts.extra_log_fields,
                    },
                )
                return result
            last_error = CommandError(
                "command failed",
                argv=full,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        if attempt <= opts.retries:
            log.warning(
                "command failed, retrying",
                extra={
                    "argv": redact(shlex.join(full)),
                    "attempt": attempt,
                    "of": opts.retries + 1,
                    **opts.extra_log_fields,
                },
            )
            time.sleep(opts.retry_delay_s)

    assert last_error is not None
    raise last_error


def run_bytes(argv: Sequence[str], opts: RunOptions | None = None) -> bytes:
    """Run a command and return its stdout as raw bytes.

    :func:`run` decodes stdout as UTF-8 with ``errors="replace"``, which is
    right for logs and catastrophic for data: every byte sequence that is not
    valid UTF-8 becomes U+FFFD and is gone. Anything that produces binary —
    raw PCM out of ffmpeg, for one — has to come through here instead.
    """
    opts = opts or RunOptions()
    argv = list(argv)
    require(argv[0])
    full = _wrap_priority(argv, opts)

    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 - argv is a list, never a shell string
            full,
            check=False,
            capture_output=True,
            timeout=opts.timeout_s,
            cwd=str(opts.cwd) if opts.cwd else None,
            env=_build_env(opts),
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandError(
            f"timed out after {opts.timeout_s:.0f}s",
            argv=full,
            stderr=_tail(_as_text(exc.stderr), opts.max_captured_lines),
        ) from None

    stderr = _tail(completed.stderr.decode("utf-8", errors="replace"), opts.max_captured_lines)
    if completed.returncode != 0 and opts.check:
        raise CommandError(
            "command failed", argv=full, returncode=completed.returncode, stderr=stderr
        )

    log.debug(
        "command finished",
        extra={
            "argv": redact(shlex.join(full)),
            "rc": completed.returncode,
            "bytes": len(completed.stdout),
            "duration_s": round(time.monotonic() - started, 3),
        },
    )
    return completed.stdout


class PipedCommand:
    """A long-running child fed through stdin, used to stream raw frames.

    :func:`run` buffers everything, which is wrong for a process that
    consumes gigabytes of pixels. This keeps the same guarantees — argv is a
    list, stderr is captured and redacted, failures raise CommandError with
    the tail of the output — while letting the caller write frames as they
    are produced.

    stderr is drained by a thread. Without that, ffmpeg blocks forever once
    the pipe buffer fills, and the generator appears to hang at a random
    frame with no error at all.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float = 3600.0,
        nice: int | None = None,
        ionice_class: str | None = None,
        max_captured_lines: int = 400,
    ) -> None:
        require(argv[0])
        opts = RunOptions(nice=nice, ionice_class=ionice_class)
        self.argv = _wrap_priority(argv, opts)
        self.timeout_s = timeout_s
        self._max_lines = max_captured_lines
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: list[str] = []
        self._reader: threading.Thread | None = None

    def __enter__(self) -> PipedCommand:
        log.debug("starting piped command", extra={"argv": redact(shlex.join(self.argv))})
        self._process = subprocess.Popen(  # noqa: S603 - argv is a list, never a shell string
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()
        return self

    def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for raw in self._process.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            self._stderr.append(line)
            if len(self._stderr) > self._max_lines:
                del self._stderr[: len(self._stderr) - self._max_lines]

    @property
    def stderr_text(self) -> str:
        return "\n".join(self._stderr)

    def write(self, payload: bytes) -> None:
        """Feed the child. A dead child surfaces as CommandError, not EPIPE."""
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(payload)
        except BrokenPipeError:
            self._process.wait(timeout=10)
            raise CommandError(
                "the process closed its input early",
                argv=self.argv,
                returncode=self._process.returncode,
                stderr=_tail(self.stderr_text, self._max_lines),
            ) from None

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._process is not None
        process = self._process

        if exc_type is not None:
            # Our own failure: stop the child rather than let it wait on a
            # stdin that will never be written to again.
            with contextlib.suppress(OSError):
                if process.stdin:
                    process.stdin.close()
            with contextlib.suppress(Exception):
                process.kill()
                process.wait(timeout=10)
            return False

        with contextlib.suppress(OSError, BrokenPipeError):
            if process.stdin:
                process.stdin.close()

        try:
            returncode = process.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            raise CommandError(
                f"timed out after {self.timeout_s:.0f}s",
                argv=self.argv,
                stderr=_tail(self.stderr_text, self._max_lines),
            ) from None

        if self._reader is not None:
            self._reader.join(timeout=5)

        if returncode != 0:
            raise CommandError(
                "command failed",
                argv=self.argv,
                returncode=returncode,
                stderr=_tail(self.stderr_text, self._max_lines),
            )
        return False


async def run_async(argv: Sequence[str], opts: RunOptions | None = None) -> CommandResult:
    """Same contract as :func:`run`, without blocking the event loop."""
    return await asyncio.to_thread(run, argv, opts)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tail(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    dropped = len(lines) - max_lines
    return f"[... {dropped} earlier lines omitted ...]\n" + "\n".join(lines[-max_lines:])


async def terminate(  # noqa: PLR0911 - each exit is a distinct outcome worth naming
    process: asyncio.subprocess.Process | subprocess.Popen[Any],
    *,
    grace_s: float = 10.0,
) -> int | None:
    """Stop a long-running child politely, then forcibly.

    ffmpeg needs SIGINT (not SIGTERM) to flush and close the RTMP session
    cleanly; killing it outright leaves the YouTube ingest hanging until it
    times out on its own.
    """
    if process.returncode is not None:
        return process.returncode

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            return process.returncode
        except (OSError, ValueError):
            break
        try:
            if isinstance(process, asyncio.subprocess.Process):
                return await asyncio.wait_for(process.wait(), timeout=grace_s)
            return await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=grace_s)
        except TimeoutError:
            log.warning("process ignored %s, escalating", sig.name)

    try:
        process.kill()
    except ProcessLookupError:
        return process.returncode
    if isinstance(process, asyncio.subprocess.Process):
        return await process.wait()
    return await asyncio.to_thread(process.wait)


__all__ = [
    "CommandError",
    "CommandResult",
    "RunOptions",
    "require",
    "run",
    "run_async",
    "run_bytes",
    "terminate",
    "which",
]
