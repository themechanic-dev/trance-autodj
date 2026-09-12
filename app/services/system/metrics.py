"""Live system metrics for the dashboard.

Deliberately cheap: this is polled every few seconds by a browser, so it must
cost far less than the work it is reporting on.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core import hardware


@dataclass(frozen=True)
class Metrics:
    cpu_percent: float
    cpu_count: int
    load_1m: float
    load_5m: float
    load_15m: float
    memory_total_mb: int
    memory_used_mb: int
    memory_percent: float
    disk_total_gb: float
    disk_free_gb: float
    disk_percent: float
    temperature_c: float | None
    gpu_present: bool
    gpu_name: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _temperature() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        sensors = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None
    # Prefer a package/CPU sensor, otherwise the hottest reading we can see.
    for key in ("k10temp", "coretemp", "cpu_thermal", "acpitz"):
        for entry in sensors.get(key, ()):
            if entry.current:
                return float(entry.current)
    readings = [e.current for group in sensors.values() for e in group if e.current]
    return float(max(readings)) if readings else None


def collect(data_dir: Path, *, cpu_interval: float = 0.0) -> Metrics:
    """Sample the machine.

    ``cpu_interval`` of 0 makes psutil report since the previous call, which is
    what a polling dashboard wants; a blocking interval would add its own load.
    """
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    load = hardware.load_average()
    count = hardware.cpu_count()

    if psutil is not None:
        cpu_percent = float(psutil.cpu_percent(interval=cpu_interval or None))
        memory = psutil.virtual_memory()
        memory_total_mb = int(memory.total / 1024**2)
        memory_used_mb = int((memory.total - memory.available) / 1024**2)
        memory_percent = float(memory.percent)
    else:
        # Without psutil, approximate CPU load from the 1-minute average.
        cpu_percent = min(100.0, load[0] / count * 100.0)
        memory_total_mb = memory_used_mb = 0
        memory_percent = 0.0

    usage = shutil.disk_usage(data_dir)
    gpu = hardware.gpu_info()

    return Metrics(
        cpu_percent=round(cpu_percent, 1),
        cpu_count=count,
        load_1m=round(load[0], 2),
        load_5m=round(load[1], 2),
        load_15m=round(load[2], 2),
        memory_total_mb=memory_total_mb,
        memory_used_mb=memory_used_mb,
        memory_percent=round(memory_percent, 1),
        disk_total_gb=round(usage.total / 1024**3, 1),
        disk_free_gb=round(usage.free / 1024**3, 1),
        disk_percent=round(usage.used / usage.total * 100.0, 1) if usage.total else 0.0,
        temperature_c=_temperature(),
        gpu_present=gpu.present,
        gpu_name=gpu.name,
    )


__all__ = ["Metrics", "collect"]
