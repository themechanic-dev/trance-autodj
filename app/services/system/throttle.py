"""When the generator is allowed to work.

Requirement #1 of the brief is that the CPU is not pinned. Three independent
brakes, in order of authority:

1. **systemd** — ``Nice=19`` and ``CPUQuota=`` in the unit file. The kernel
   enforces this whatever the process believes, which is why it comes first.
2. **This module** — refuse to *start* a job while the machine is already
   busy. Cheap, and it is what keeps an interactive session responsive.
3. **The generator itself** — one job at a time, no internal parallelism.

The load average is checked before a job, not during it. A half-built block
is wasted work, so once started it runs to completion; the brake is on
starting, not on finishing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from app.core import hardware
from app.core.config import Config


@dataclass(frozen=True)
class ThrottleDecision:
    allowed: bool
    reason: str
    load_1m: float
    threshold: float
    quiet_hours: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "load_1m": round(self.load_1m, 2),
            "threshold": round(self.threshold, 2),
            "quiet_hours": self.quiet_hours,
        }


def _parse_hhmm(value: str) -> time:
    hours, _, minutes = value.partition(":")
    return time(int(hours), int(minutes))


def in_quiet_hours(cfg: Config, now: datetime | None = None) -> bool:
    """Is the clock inside the configured full-speed window?

    Handles windows that cross midnight (02:00-08:00 does not, 22:00-06:00
    does), which is the case people actually configure.
    """
    quiet = cfg.cpu.quiet_hours
    if not quiet.enabled:
        return False
    now = now or datetime.now()
    start, end = _parse_hhmm(quiet.start), _parse_hhmm(quiet.end)
    current = now.time()
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def evaluate(
    cfg: Config, *, streaming: bool = False, now: datetime | None = None
) -> ThrottleDecision:
    """Decide whether a new generation job may start right now."""
    load_1m = hardware.load_average()[0]
    threshold = hardware.cpu_count() * cfg.cpu.load_threshold_factor
    quiet = in_quiet_hours(cfg, now)

    if streaming and cfg.cpu.pause_during_stream:
        return ThrottleDecision(
            allowed=False,
            reason="generation is paused while the stream is live (cpu.pause_during_stream)",
            load_1m=load_1m,
            threshold=threshold,
            quiet_hours=quiet,
        )

    if quiet:
        # Inside quiet hours the machine is ours; ignore the load brake.
        return ThrottleDecision(
            allowed=True,
            reason="quiet hours: running at full speed",
            load_1m=load_1m,
            threshold=threshold,
            quiet_hours=True,
        )

    if load_1m > threshold:
        return ThrottleDecision(
            allowed=False,
            reason=f"load average {load_1m:.2f} is above the threshold {threshold:.2f}",
            load_1m=load_1m,
            threshold=threshold,
            quiet_hours=False,
        )

    return ThrottleDecision(
        allowed=True,
        reason="the machine is idle enough",
        load_1m=load_1m,
        threshold=threshold,
        quiet_hours=False,
    )


__all__ = ["ThrottleDecision", "evaluate", "in_quiet_hours"]
