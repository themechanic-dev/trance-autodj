"""Supervision of the Icecast and Liquidsoap child processes.

These processes hold ports. A second copy of either cannot start while the
first is alive, so every mistake here shows up as a restart loop that runs for
the lifetime of the container.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path

import pytest

from app.core.config import load_config
from app.core.paths import Paths
from app.core.runtime import Runtime
from app.core.security import SecretStore
from app.services.audio.manager import AudioManager
from app.services.audio.process import SupervisedProcess

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _supervisors(name: str) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == f"{name}-supervisor" and t.is_alive()]


def _wait_until(predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture
def sleeper(tmp_path: Path) -> Path:
    return _script(tmp_path / "sleeper", "exec sleep 300")


# -- the process object ----------------------------------------------------


def test_stop_ends_the_supervisor_thread(sleeper: Path):
    """A supervisor left running after stop() is unreachable forever: nothing
    holds a reference to it and it keeps restarting a child on its own."""
    process = SupervisedProcess(name="sleeper", argv=[str(sleeper)])
    process.start()
    assert _wait_until(lambda: len(_supervisors("sleeper")) == 1)
    process.stop()
    assert _supervisors("sleeper") == []


def test_restart_leaves_exactly_one_supervisor(sleeper: Path):
    """restart() is stop() then start(), and start() clears the flag stop()
    just set. Get the ordering wrong and you end up with either two restart
    loops fighting over one port, or none at all."""
    process = SupervisedProcess(name="sleeper", argv=[str(sleeper)])
    process.start()
    assert _wait_until(lambda: len(_supervisors("sleeper")) == 1)
    for _ in range(3):
        process.restart()
        assert len(_supervisors("sleeper")) == 1, "restart changed the number of supervisors"
    try:
        assert process.running
    finally:
        process.stop()


def test_the_child_is_still_supervised_after_restart(sleeper: Path):
    """The other way to lose this race: stop() sets the flag, the old loop is
    still winding down when start() checks whether a supervisor exists, so
    start() declines to create one and the new child is left with nobody
    watching it. Nothing looks wrong until it dies and never comes back."""
    process = SupervisedProcess(name="sleeper", argv=[str(sleeper)])
    process.start()
    try:
        for _ in range(5):
            process.restart()
        before = process.status().restarts
        pid = process.status().pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        assert _wait_until(
            lambda: process.status().restarts > before
        ), "the child died and was never restarted"
        assert _wait_until(lambda: process.running)
    finally:
        process.stop()


def test_start_twice_does_not_spawn_a_second_child(sleeper: Path):
    process = SupervisedProcess(name="sleeper", argv=[str(sleeper)])
    process.start()
    first = process.status().pid
    process.start()
    try:
        assert process.status().pid == first
        assert len(_supervisors("sleeper")) == 1
    finally:
        process.stop()


def test_a_child_that_dies_is_still_restarted(tmp_path: Path):
    """The fix must not buy a quiet log by giving up supervision."""
    quitter = _script(tmp_path / "quitter", "exit 3")
    process = SupervisedProcess(name="quitter", argv=[str(quitter)])
    process.start()
    try:
        assert _wait_until(lambda: process.status().restarts >= 2, timeout_s=15.0)
    finally:
        process.stop()


# -- the manager -----------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path, monkeypatch) -> AudioManager:
    icecast = _script(tmp_path / "fake-icecast", "exec sleep 300")
    liquidsoap = _script(tmp_path / "fake-liquidsoap", "exec sleep 300")
    cfg = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "data"),
            "TAD_APP__CONFIG_DIR": str(CONFIG_DIR),
            "TAD_AUDIO__ICECAST__BINARY": str(icecast),
            "TAD_AUDIO__LIQUIDSOAP__BINARY": str(liquidsoap),
        },
    )
    paths = Paths.from_config(cfg)
    paths.ensure()
    runtime = Runtime(
        config=cfg,
        paths=paths,
        secrets=SecretStore.open(paths.secrets_file),
        service="test",
    )
    # The fake Icecast never answers HTTP; waiting for it would only add the
    # timeout to every test in this file.
    monkeypatch.setattr(AudioManager, "_wait_for_icecast", lambda self: True)
    audio = AudioManager(runtime)
    yield audio
    audio.stop()


def test_start_twice_keeps_the_running_children(manager: AudioManager):
    """The original fault: start() rebuilt both process objects every time, so
    the second press orphaned two running children together with their
    supervisors, and the replacements then failed to bind the ports the
    orphans still held — once per press, for as long as the container ran."""
    manager.start()
    pids = (manager._icecast.status().pid, manager._liquidsoap.status().pid)
    manager.start()
    assert (manager._icecast.status().pid, manager._liquidsoap.status().pid) == pids
    assert len(_supervisors("icecast")) == 1
    assert len(_supervisors("liquidsoap")) == 1


def test_stop_then_start_runs_again(manager: AudioManager):
    manager.start()
    manager.stop()
    assert not manager.status().icecast_running
    manager.start()
    assert manager.status().icecast_running
    assert manager.status().liquidsoap_running
    assert len(_supervisors("liquidsoap")) == 1


def test_a_changed_binary_stops_the_old_child(manager: AudioManager, tmp_path: Path):
    manager.start()
    old = manager._liquidsoap
    manager.cfg.audio.liquidsoap.binary = str(_script(tmp_path / "other", "exec sleep 300"))
    manager.start()
    assert manager._liquidsoap is not old
    assert not old.running, "the replaced child was left running and unreachable"
    assert len(_supervisors("liquidsoap")) == 1


# -- surviving a restart ---------------------------------------------------


def test_a_fresh_install_does_not_come_up_on_air(manager: AudioManager):
    """Nothing has been asked for yet, so nothing should start by itself."""
    assert not manager.was_on_air()


def test_being_on_air_survives_the_process(manager: AudioManager):
    """The container restarts itself on failure and on boot. Without a
    remembered intent it came back silent and off air while reporting perfect
    health, until somebody happened to open a browser."""
    manager.start()
    assert manager.was_on_air()

    # A new manager over the same data directory is what a restart looks like.
    successor = AudioManager(manager.runtime)
    assert successor.was_on_air()


def test_a_deliberate_stop_is_remembered_too(manager: AudioManager):
    """Resume restores intent, not the last observed state: an operator who
    stopped the audio should not find it running again after a reboot."""
    manager.start()
    manager.stop()
    assert not manager.was_on_air()
    assert not AudioManager(manager.runtime).was_on_air()


def test_a_restart_ends_up_on_air(manager: AudioManager):
    manager.start()
    manager.restart()
    assert manager.was_on_air()


def test_shutting_down_does_not_forget_that_we_were_on_air(manager: AudioManager):
    """The bug that made resume_on_start do nothing at all: the shutdown path
    called stop(), stop() recorded "off air", and so every clean restart
    erased the very intent it was about to look for."""
    manager.start()
    manager.stop(remember=False)
    assert manager.was_on_air()
    assert AudioManager(manager.runtime).was_on_air()


def test_starting_the_audio_forgets_the_previous_run(manager: AudioManager):
    """Now-playing is written by Liquidsoap at each track boundary, which on a
    trance station can be six minutes away. Until then the dashboard used to
    show whatever was playing last time — a track that may not even be in the
    library any more."""
    manager.paths.nowplaying_file.parent.mkdir(parents=True, exist_ok=True)
    manager.paths.nowplaying_file.write_text('{"title": "from a previous life"}', encoding="utf-8")
    manager.start()
    assert not manager.paths.nowplaying_file.exists()
