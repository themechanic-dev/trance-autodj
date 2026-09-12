"""The preflight report must be complete, honest and machine-readable."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.config import Config, load_config
from app.core.paths import Paths
from app.services.system.preflight import Status, format_report, run_preflight


def test_report_covers_every_category(config: Config, paths: Paths):
    report = run_preflight(config, paths)
    assert set(report.by_category()) >= {"config", "runtime", "tools", "storage", "hardware"}


def test_report_serialises(config: Config, paths: Paths):
    payload = run_preflight(config, paths).as_dict()
    assert isinstance(payload["ok"], bool)
    assert payload["checks"]
    for check in payload["checks"]:
        assert check["status"] in {"ok", "warn", "fail"}
        assert check["key"] and check["title"]


def test_missing_tools_are_a_failure_not_a_crash(tmp_path: Path, monkeypatch):
    """On a machine with no ffmpeg the report must still be produced."""
    monkeypatch.setenv("PATH", "")
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "d")})
    report = run_preflight(cfg)
    assert not report.ok
    keys = {c.key for c in report.failures}
    assert {"ffmpeg", "ffprobe", "liquidsoap"} <= keys
    # Every failure explains what to do about it.
    for check in report.failures:
        assert check.hint


def test_directories_are_created_by_the_check(tmp_path: Path):
    target = tmp_path / "fresh"
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(target)})
    paths = Paths.from_config(cfg)
    assert not target.exists()
    run_preflight(cfg, paths)
    assert paths.blocks.is_dir()
    assert paths.state.is_dir()


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses file permissions, so 0500 is still writable"
)
def test_unwritable_data_dir_fails_clearly(tmp_path: Path):
    root = tmp_path / "ro"
    root.mkdir()
    root.chmod(0o500)
    try:
        cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(root)})
        report = run_preflight(cfg)
        failures = {c.key for c in report.failures}
        assert failures & {"data_writable"} or any(k.startswith("dir:") for k in failures)
    finally:
        root.chmod(0o700)


def test_copy_mode_is_reported_as_the_good_path(config: Config, paths: Paths):
    check = next(c for c in run_preflight(config, paths).checks if c.key == "video_mode")
    assert check.status is Status.OK
    assert "never re-encoded" in check.detail


def test_reencode_mode_warns(tmp_path: Path):
    cfg = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "d"),
            "TAD_STREAM__VIDEO_MODE": "reencode",
        },
    )
    check = next(c for c in run_preflight(cfg).checks if c.key == "video_mode")
    assert check.status is Status.WARN


def test_format_report_is_plain_text_without_colour(config: Config, paths: Paths):
    text = format_report(run_preflight(config, paths), color=False)
    assert "\033[" not in text
    assert "HARDWARE" in text
