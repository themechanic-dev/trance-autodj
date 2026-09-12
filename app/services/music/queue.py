"""The composing queue: as many tracks as anyone asks for, at the pace the
machine can manage, without the broadcast noticing.

A batch of a hundred tracks is a day's work on a Ryzen and a week on a NAS,
and neither can happen inside an HTTP request or a thread of the web server.
So the queue lives in the database — one job row per track, which survives
a restart — and one worker thread takes the oldest waiting job, composes it
in a process of its own, files the result, and looks for the next.

A process rather than a thread, three reasons: it can be niced below
everything else so the stream always wins the CPU; it can be killed cleanly
when the operator changes their mind; and if numpy ever falls over on some
odd seed it takes one track with it, not the dashboard.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading

from sqlmodel import Session, select

from app.core.logging import get_logger
from app.core.runtime import Runtime
from app.models.entities import GenerationJob, JobKind, JobStatus, Track, utcnow
from app.services.audio import library, playlists

log = get_logger(__name__)

#: How long one track may take before we decide the process has hung. A
#: five-minute track is half a minute on a desktop and perhaps six on a slow
#: NAS; an hour is nowhere near either and still catches a real hang.
TRACK_TIMEOUT_S = 3600.0

#: How often the worker looks for work when there is none.
IDLE_POLL_S = 5.0

#: Lowest priority. The broadcast, the audio stack and the dashboard all
#: come first; composing gets whatever is left, which on a NAS that is also
#: streaming may be very little — and that is correct.
NICENESS = 19


def enqueue(
    session: Session, *, seeds: list[int], minutes: float, style: str, playlist: str
) -> list[int]:
    """Add tracks to the queue. Nothing is composed here."""
    ids = []
    for position, seed in enumerate(seeds, start=1):
        job = GenerationJob(
            kind=JobKind.TRACK,
            status=JobStatus.QUEUED,
            step=f"waiting ({position}/{len(seeds)})",
            params_json=json.dumps(
                {"seed": seed, "minutes": minutes, "style": style, "playlist": playlist}
            ),
        )
        session.add(job)
        session.flush()
        ids.append(job.id)
    return ids


def cancel_waiting(session: Session) -> int:
    """Take everything not yet started out of the queue. The track being
    composed right now finishes; the rest never begin."""
    waiting = session.exec(
        select(GenerationJob)
        .where(GenerationJob.kind == JobKind.TRACK)
        .where(GenerationJob.status == JobStatus.QUEUED)
    ).all()
    for job in waiting:
        job.status = JobStatus.CANCELLED
        job.step = "cancelled"
        job.finished_at = utcnow()
    return len(waiting)


def requeue_interrupted(session: Session) -> int:
    """Jobs left RUNNING by a process that is no longer running were
    interrupted — a restart, a crash — and go back to the front of the
    queue rather than sitting at 'running' forever."""
    stuck = session.exec(
        select(GenerationJob)
        .where(GenerationJob.kind == JobKind.TRACK)
        .where(GenerationJob.status == JobStatus.RUNNING)
    ).all()
    for job in stuck:
        job.status = JobStatus.QUEUED
        job.step = "interrupted — will retry"
        job.started_at = None
    return len(stuck)


def waiting_count(session: Session) -> int:
    return len(
        session.exec(
            select(GenerationJob.id)
            .where(GenerationJob.kind == JobKind.TRACK)
            .where(GenerationJob.status == JobStatus.QUEUED)
        ).all()
    )


def _next_job(session: Session) -> GenerationJob | None:
    return session.exec(
        select(GenerationJob)
        .where(GenerationJob.kind == JobKind.TRACK)
        .where(GenerationJob.status == JobStatus.QUEUED)
        .order_by(GenerationJob.created_at)  # type: ignore[arg-type]
    ).first()


def compose_in_subprocess(
    runtime: Runtime, *, seed: int, minutes: float, style: str, timeout_s: float = TRACK_TIMEOUT_S
) -> dict:
    """Run the renderer as a niced child and hand back what it printed."""
    argv = [
        sys.executable,
        "-m",
        "app.services.music.render",
        "--music-dir",
        str(runtime.paths.music),
        "--seed",
        str(seed),
        "--minutes",
        str(minutes),
        "--style",
        style,
    ]

    def lower_priority() -> None:  # pragma: no cover - runs in the child
        with contextlib.suppress(OSError):
            os.nice(NICENESS)

    result = subprocess.run(  # noqa: S603 - argv is a list we built
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        preexec_fn=lower_priority,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-800:] or f"exit code {result.returncode}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def file_result(session: Session, runtime: Runtime, made: dict, playlist: str) -> None:
    """Put a finished track where the station will find it."""
    library.scan(session, runtime.paths)
    track = session.exec(select(Track).where(Track.relpath == made["relpath"])).first()
    if track is not None:
        # We composed it, so the tempo is known exactly rather than estimated:
        # no analysis pass, and the beat-aligned crossfade gets a number it
        # can trust.
        track.bpm = round(float(made["bpm"]), 2)
        track.bpm_confidence = 1.0
    if playlist:
        from app.api.audio import _put_in_playlist

        _put_in_playlist(session, runtime, playlist, [made["relpath"]])
    playlists.ensure_default(session, runtime.paths, runtime.config)


class ComposerWorker:
    """One thread, one track at a time, for as long as there is a queue."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.current_seed: int | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="composer", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        from app.core.db import session_scope

        with session_scope() as session:
            resumed = requeue_interrupted(session)
        if resumed:
            log.info("composing resumes after a restart", extra={"requeued": resumed})

        while not self._stop.is_set():
            if not self.run_once():
                self._stop.wait(IDLE_POLL_S)

    def run_once(self) -> bool:
        """Compose one waiting track. Returns whether there was one."""
        from app.core.db import session_scope

        with session_scope() as session:
            job = _next_job(session)
            if job is None:
                return False
            job.status = JobStatus.RUNNING
            job.started_at = utcnow()
            params = json.loads(job.params_json or "{}")
            job.step = f"composing seed {params.get('seed')} · {params.get('style', 'mixed')}"
            job_id = job.id

        seed = int(params.get("seed", 0))
        self.current_seed = seed
        error, made = "", None
        try:
            made = compose_in_subprocess(
                self.runtime,
                seed=seed,
                minutes=float(params.get("minutes", 5.0)),
                style=str(params.get("style", "mixed")),
            )
            with session_scope() as session:
                file_result(session, self.runtime, made, str(params.get("playlist", "")))
            status = JobStatus.DONE
            log.info("track composed", extra={"seed": seed, "title": made["title"]})
        except subprocess.TimeoutExpired:
            status, error = JobStatus.FAILED, f"gave up after {TRACK_TIMEOUT_S:.0f}s"
            log.error("composing seed %d timed out", seed)
        except Exception as exc:
            status, error = JobStatus.FAILED, str(exc)[:2000]
            log.exception("composing seed %d failed", seed)
        finally:
            self.current_seed = None

        with session_scope() as session:
            job = session.get(GenerationJob, job_id)
            if job is not None:
                job.status = status
                job.progress = 1.0
                job.step = f"{made['title']} · {made['style']}" if made else status.value
                job.result_id = str(seed)
                job.error = error
                job.finished_at = utcnow()
        return True


__all__ = [
    "IDLE_POLL_S",
    "NICENESS",
    "TRACK_TIMEOUT_S",
    "ComposerWorker",
    "cancel_waiting",
    "compose_in_subprocess",
    "enqueue",
    "file_result",
    "requeue_interrupted",
    "waiting_count",
]
