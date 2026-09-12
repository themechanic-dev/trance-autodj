"""The composing queue: as many as asked, survives a restart, never outranks
the broadcast."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlmodel import select

from app.core.config import load_config
from app.core.db import create_tables, dispose, init_engine, session_scope
from app.core.paths import Paths
from app.core.runtime import Runtime
from app.core.security import SecretStore
from app.models.entities import GenerationJob, JobStatus, Track
from app.services.music import queue


@pytest.fixture
def runtime(tmp_path: Path):
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "data")})
    paths = Paths.from_config(cfg)
    paths.ensure()
    init_engine(paths.db_file)
    create_tables()
    yield Runtime(
        config=cfg, paths=paths, secrets=SecretStore.open(paths.secrets_file), service="test"
    )
    dispose()


def _jobs():
    with session_scope() as session:
        return [
            (j.status, json.loads(j.params_json)["seed"])
            for j in session.exec(
                select(GenerationJob).order_by(GenerationJob.created_at)  # type: ignore[arg-type]
            ).all()
        ]


def test_asking_for_a_hundred_costs_nothing_up_front(runtime):
    """The request returns at once; the work is rows in a table."""
    with session_scope() as session:
        ids = queue.enqueue(
            session, seeds=list(range(100)), minutes=5.0, style="mixed", playlist="Generated"
        )
    assert len(ids) == 100
    with session_scope() as session:
        assert queue.waiting_count(session) == 100
    assert (
        not list((runtime.paths.music / "generated").glob("*.mp3"))
        if (runtime.paths.music / "generated").exists()
        else True
    )


def test_cancelling_drops_the_waiting_and_spares_the_running(runtime):
    with session_scope() as session:
        queue.enqueue(session, seeds=[1, 2, 3], minutes=1.0, style="mixed", playlist="")
        first = session.exec(select(GenerationJob)).first()
        first.status = JobStatus.RUNNING
    with session_scope() as session:
        assert queue.cancel_waiting(session) == 2
    statuses = [status for status, _ in _jobs()]
    assert statuses == [JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.CANCELLED]


def test_a_job_interrupted_by_a_restart_goes_back_in_the_queue(runtime):
    """A container that restarts mid-track must not leave the row saying
    'running' forever with nothing running it."""
    with session_scope() as session:
        queue.enqueue(session, seeds=[7], minutes=1.0, style="mixed", playlist="")
        job = session.exec(select(GenerationJob)).first()
        job.status = JobStatus.RUNNING
    with session_scope() as session:
        assert queue.requeue_interrupted(session) == 1
        assert queue.waiting_count(session) == 1


def test_the_oldest_waits_least(runtime):
    with session_scope() as session:
        queue.enqueue(session, seeds=[11, 22, 33], minutes=1.0, style="mixed", playlist="")
    with session_scope() as session:
        assert json.loads(queue._next_job(session).params_json)["seed"] == 11


def test_one_track_goes_all_the_way_through(runtime):
    """The real thing: a niced child process composes it, and the worker files
    it in the library with the tempo it was composed at."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is what encodes the mp3")

    with session_scope() as session:
        queue.enqueue(session, seeds=[99], minutes=1.0, style="uplifting", playlist="Made")

    worker = queue.ComposerWorker(runtime)
    assert worker.run_once() is True
    assert worker.run_once() is False, "there was only one"

    assert _jobs() == [(JobStatus.DONE, 99)]
    with session_scope() as session:
        track = session.exec(select(Track)).first()
        assert track is not None
        assert track.relpath.startswith("generated/")
        assert track.bpm and track.bpm_confidence == 1.0
        from app.models.entities import Playlist

        made = session.exec(select(Playlist).where(Playlist.name == "Made")).first()
        assert made is not None and len(made.items) == 1


def test_a_failing_track_does_not_stop_the_queue(runtime, monkeypatch):
    """One odd seed must cost one track, not the batch."""
    calls = []

    def explode(*_a, **kw):
        calls.append(kw["seed"])
        if kw["seed"] == 2:
            raise RuntimeError("numpy fell over")
        return {
            "relpath": f"generated/{kw['seed']}.mp3",
            "title": "t",
            "bpm": 138.0,
            "key": "A minor",
            "seed": kw["seed"],
            "duration_s": 60.0,
            "style": "uplifting",
        }

    monkeypatch.setattr(queue, "compose_in_subprocess", explode)
    monkeypatch.setattr(queue, "file_result", lambda *a, **k: None)

    with session_scope() as session:
        queue.enqueue(session, seeds=[1, 2, 3], minutes=1.0, style="mixed", playlist="")
    worker = queue.ComposerWorker(runtime)
    while worker.run_once():
        pass

    assert calls == [1, 2, 3]
    assert [s for s, _ in _jobs()] == [JobStatus.DONE, JobStatus.FAILED, JobStatus.DONE]
    with session_scope() as session:
        failed = session.exec(
            select(GenerationJob).where(GenerationJob.status == JobStatus.FAILED)
        ).first()
        assert "numpy fell over" in failed.error


def test_the_child_runs_at_the_lowest_priority():
    """The broadcast comes first. Whatever else changes, this must not."""
    assert queue.NICENESS == 19


def test_the_worker_starts_and_stops(runtime):
    worker = queue.ComposerWorker(runtime)
    worker.start()
    assert worker.running
    worker.stop()
    assert not worker.running
