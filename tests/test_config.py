"""Configuration loading: defaults, YAML, environment, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Config, env_overrides, load_config


def test_defaults_are_valid_without_any_file():
    cfg = load_config("/nonexistent/config.yaml", environ={})
    assert cfg.app.port == 8080
    assert cfg.audio.crossfade.duration_s == 14.0
    assert cfg.video.width == 1280
    assert cfg.stream.video_mode == "copy"
    assert cfg.source_path is None


def test_yaml_file_overrides_defaults(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "app:\n  port: 9999\naudio:\n  crossfade:\n    duration_s: 20\n",
        encoding="utf-8",
    )
    cfg = load_config(path, environ={})
    assert cfg.app.port == 9999
    assert cfg.audio.crossfade.duration_s == 20.0
    # Untouched values keep their defaults.
    assert cfg.video.fps == 30
    assert cfg.source_path == path


def test_environment_beats_the_file(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("app:\n  port: 9999\n", encoding="utf-8")
    cfg = load_config(path, environ={"TAD_APP__PORT": "7777"})
    assert cfg.app.port == 7777


def test_env_overrides_builds_nested_dicts():
    result = env_overrides(
        {
            "TAD_AUDIO__CROSSFADE__DURATION_S": "18.5",
            "TAD_APP__LOG_LEVEL": "debug",
            "PATH": "/usr/bin",
        }
    )
    assert result == {
        "audio": {"crossfade": {"duration_s": 18.5}},
        "app": {"log_level": "debug"},
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("False", False),
        ("off", False),
        ("42", 42),
        ("3.5", 3.5),
        ('["a", "b"]', ["a", "b"]),
        ("plain text", "plain text"),
    ],
)
def test_env_scalar_coercion(raw: str, expected: object):
    assert env_overrides({"TAD_X": raw})["x"] == expected


def test_process_level_env_vars_are_not_treated_as_settings():
    """TAD_MASTER_KEY and friends address the process, not the config.

    They are documented in docker-compose.yml and set by the systemd units,
    so letting them reach the validator would turn a normal deployment into a
    startup failure on an unknown field.
    """
    cfg = load_config(
        None,
        environ={
            "TAD_MASTER_KEY": "some-fernet-key",
            "TAD_SERVICE": "generator",
            "TAD_CONFIG_FILE": "/etc/somewhere/config.yaml",
            "TAD_APP__PORT": "8123",
        },
    )
    assert cfg.app.port == 8123


def test_unknown_key_is_rejected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("app:\n  prot: 8080\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path, environ={})


def test_crossfade_duration_must_lie_within_its_own_bounds():
    with pytest.raises(ValidationError, match="must lie within"):
        Config.model_validate({"audio": {"crossfade": {"duration_s": 60}}})


def test_odd_video_dimensions_are_rejected():
    with pytest.raises(ValidationError, match="must both be even"):
        Config.model_validate({"video": {"width": 1281}})


def test_reactive_overlay_cannot_be_used_with_copy():
    with pytest.raises(ValidationError, match="reencode"):
        Config.model_validate(
            {"stream": {"video_mode": "copy", "reactive_overlay": {"enabled": True}}}
        )
    # The same setting is fine once re-encoding is accepted.
    cfg = Config.model_validate(
        {"stream": {"video_mode": "reencode", "reactive_overlay": {"enabled": True}}}
    )
    assert cfg.stream.reactive_overlay.enabled


def test_pool_bounds_are_checked():
    with pytest.raises(ValidationError, match="min_blocks"):
        Config.model_validate({"visual": {"pool": {"min_blocks": 40, "max_blocks": 10}}})


def test_quiet_hours_must_be_hhmm():
    with pytest.raises(ValidationError, match="HH:MM"):
        Config.model_validate({"cpu": {"quiet_hours": {"start": "2am"}}})


def test_profile_fingerprint_changes_with_the_profile():
    a = Config().video.profile_fingerprint()
    other = Config.model_validate({"video": {"gop": 120, "keyint_min": 120}})
    b = other.video.profile_fingerprint()
    assert a != b
    # ...and is stable for the same settings.
    assert a == Config().video.profile_fingerprint()


def test_the_shipped_example_config_is_valid():
    example = Path(__file__).resolve().parent.parent / "config" / "config.example.yaml"
    cfg = load_config(example, environ={})
    assert cfg.audio.crossfade.duration_s == 14.0
    assert cfg.visual.block.duration_s == 600.0
    assert "flowfield" in cfg.visual.procedural.generators
