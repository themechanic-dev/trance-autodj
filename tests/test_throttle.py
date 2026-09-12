"""The CPU brake: when generation is allowed to start."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.core import hardware
from app.core.config import Config
from app.services.system import throttle


@pytest.fixture
def four_cores(monkeypatch):
    monkeypatch.setattr(hardware, "cpu_count", lambda: 4)


def set_load(monkeypatch, value: float) -> None:
    monkeypatch.setattr(hardware, "load_average", lambda: (value, value, value))


def test_idle_machine_is_allowed(monkeypatch, four_cores):
    set_load(monkeypatch, 0.5)
    assert throttle.evaluate(Config()).allowed


def test_busy_machine_is_refused(monkeypatch, four_cores):
    set_load(monkeypatch, 3.5)  # threshold is 4 * 0.7 = 2.8
    decision = throttle.evaluate(Config())
    assert not decision.allowed
    assert "load average" in decision.reason


def test_the_threshold_follows_the_core_count(monkeypatch, four_cores):
    set_load(monkeypatch, 2.0)
    assert throttle.evaluate(Config()).threshold == pytest.approx(2.8)


def test_pause_during_stream_only_applies_while_live(monkeypatch, four_cores):
    set_load(monkeypatch, 0.1)
    cfg = Config.model_validate({"cpu": {"pause_during_stream": True}})
    assert throttle.evaluate(cfg, streaming=False).allowed
    decision = throttle.evaluate(cfg, streaming=True)
    assert not decision.allowed
    assert "paused" in decision.reason


def test_quiet_hours_override_the_load_brake(monkeypatch, four_cores):
    set_load(monkeypatch, 99.0)
    cfg = Config.model_validate(
        {"cpu": {"quiet_hours": {"enabled": True, "start": "02:00", "end": "08:00"}}}
    )
    decision = throttle.evaluate(cfg, now=datetime(2026, 1, 1, 3, 0))
    assert decision.allowed
    assert decision.quiet_hours


def test_outside_quiet_hours_the_brake_returns(monkeypatch, four_cores):
    set_load(monkeypatch, 99.0)
    cfg = Config.model_validate(
        {"cpu": {"quiet_hours": {"enabled": True, "start": "02:00", "end": "08:00"}}}
    )
    assert not throttle.evaluate(cfg, now=datetime(2026, 1, 1, 12, 0)).allowed


@pytest.mark.parametrize(
    ("start", "end", "hour", "expected"),
    [
        ("02:00", "08:00", 3, True),
        ("02:00", "08:00", 1, False),
        ("02:00", "08:00", 8, False),
        ("02:00", "08:00", 7, True),
        # Crossing midnight is the case people actually configure.
        ("22:00", "06:00", 23, True),
        ("22:00", "06:00", 2, True),
        ("22:00", "06:00", 12, False),
        ("22:00", "06:00", 21, False),
    ],
)
def test_quiet_hours_window_including_midnight(start, end, hour, expected):
    cfg = Config.model_validate(
        {"cpu": {"quiet_hours": {"enabled": True, "start": start, "end": end}}}
    )
    assert throttle.in_quiet_hours(cfg, datetime(2026, 1, 1, hour, 0)) is expected


def test_disabled_quiet_hours_are_never_active():
    cfg = Config.model_validate(
        {"cpu": {"quiet_hours": {"enabled": False, "start": "00:00", "end": "23:59"}}}
    )
    assert not throttle.in_quiet_hours(cfg, datetime(2026, 1, 1, 12, 0))


def test_pause_during_stream_beats_quiet_hours(monkeypatch, four_cores):
    """The broadcast is the product; nothing outranks protecting it."""
    set_load(monkeypatch, 0.1)
    cfg = Config.model_validate(
        {
            "cpu": {
                "pause_during_stream": True,
                "quiet_hours": {"enabled": True, "start": "00:00", "end": "23:59"},
            }
        }
    )
    assert not throttle.evaluate(cfg, streaming=True, now=datetime(2026, 1, 1, 3, 0)).allowed
