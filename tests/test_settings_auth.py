"""Runtime settings and dashboard authentication."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.auth import MIN_PASSWORD_LENGTH, AuthState, is_public
from app.core.config import Config, load_config
from app.core.db import create_tables, dispose, init_engine, session_scope
from app.core.paths import Paths
from app.services.system import settings as settings_service


@pytest.fixture
def db(tmp_path: Path):
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "data")})
    paths = Paths.from_config(cfg)
    paths.ensure()
    init_engine(paths.db_file)
    create_tables()
    yield cfg
    dispose()


# --- settings ------------------------------------------------------------


def test_an_override_changes_the_loaded_config(db, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TAD_APP__DATA_DIR", str(tmp_path / "data"))
    with session_scope() as session:
        accepted, rejected = settings_service.set_many(
            session, {"audio.crossfade.duration_s": 20.0}
        )
        assert accepted == ["audio.crossfade.duration_s"]
        assert rejected == []
    with session_scope() as session:
        assert settings_service.apply(session).audio.crossfade.duration_s == 20.0


def test_a_value_that_would_break_the_config_is_refused(db):
    with session_scope() as session:
        accepted, rejected = settings_service.set_many(
            session, {"audio.crossfade.duration_s": 99.0}
        )
    assert accepted == []
    assert any("must lie within" in r for r in rejected)


def test_nothing_is_written_when_validation_fails(db):
    """A rejected batch must leave the database untouched, not half-applied."""
    with session_scope() as session:
        settings_service.set_many(session, {"audio.crossfade.duration_s": 99.0, "video.fps": 25})
    with session_scope() as session:
        assert settings_service.read_overrides(session) == {}


def test_paths_outside_the_whitelist_are_refused(db):
    with session_scope() as session:
        accepted, rejected = settings_service.set_many(session, {"app.data_dir": "/etc"})
    assert accepted == []
    assert "not an editable setting" in rejected[0]


def test_the_environment_wins_over_a_stored_override(db, monkeypatch):
    """An operator pinning a value in the environment made a deployment
    decision; a web form must not quietly undo it."""
    with session_scope() as session:
        settings_service.set_many(session, {"video.fps": 25})

    monkeypatch.setenv("TAD_VIDEO__FPS", "60")
    with session_scope() as session:
        assert settings_service.apply(session).video.fps == 60
        assert "video.fps" in settings_service.environment_locked()


def test_a_locked_setting_cannot_be_written(db, monkeypatch):
    monkeypatch.setenv("TAD_VIDEO__FPS", "60")
    with session_scope() as session:
        accepted, rejected = settings_service.set_many(session, {"video.fps": 25})
    assert accepted == []
    assert "pinned by" in rejected[0]


def test_reset_removes_an_override(db):
    with session_scope() as session:
        settings_service.set_many(session, {"video.bitrate_k": 5000})
    with session_scope() as session:
        assert settings_service.reset(session, "video.bitrate_k") is True
    with session_scope() as session:
        assert settings_service.read_overrides(session) == {}
        assert settings_service.reset(session, "video.bitrate_k") is False


def test_describe_covers_every_editable_path(db):
    with session_scope() as session:
        rows = settings_service.describe(session, Config())
    assert {r["path"] for r in rows} == set(settings_service.EDITABLE)
    for row in rows:
        assert row["label"] and row["group"]
        assert row["value"] is not None, row["path"]


def test_audio_changes_are_flagged_for_a_restart():
    assert settings_service.needs_audio_restart(["audio.crossfade.duration_s"])
    assert not settings_service.needs_audio_restart(["video.fps", "visual.pool.max_blocks"])


# --- authentication ------------------------------------------------------


def auth_state(**overrides) -> AuthState:
    cfg = Config.model_validate({"auth": {"session_secret": "test-secret", **overrides}})
    return AuthState.from_config(cfg)


def test_a_fresh_install_has_no_password():
    assert not auth_state().configured


def test_password_round_trip():
    auth = auth_state()
    auth.set_password("a-good-password")
    assert auth.configured
    assert auth.check("admin", "a-good-password")
    assert not auth.check("admin", "wrong")
    assert not auth.check("someone", "a-good-password")


def test_short_passwords_are_refused():
    with pytest.raises(ValueError, match="at least"):
        auth_state().set_password("x" * (MIN_PASSWORD_LENGTH - 1))


def test_a_session_token_round_trips():
    auth = auth_state()
    auth.set_password("a-good-password")
    assert auth.valid(auth.issue())


def test_a_tampered_token_is_rejected():
    auth = auth_state()
    auth.set_password("a-good-password")
    token = auth.issue()
    assert not auth.valid(token[:-3] + "aaa")
    assert not auth.valid("")
    assert not auth.valid(None)


def test_a_token_from_another_secret_is_rejected():
    first = auth_state()
    first.set_password("a-good-password")
    token = first.issue()

    other = AuthState.from_config(
        Config.model_validate({"auth": {"session_secret": "different-secret"}})
    )
    other.set_password("a-good-password")
    assert not other.valid(token)


def test_an_expired_token_is_rejected():
    auth = auth_state(session_max_age_s=60)
    auth.set_password("a-good-password")
    token = auth.issue()
    auth.max_age_s = -1  # everything is now in the past
    assert not auth.valid(token)


@pytest.mark.parametrize(
    ("path", "public"),
    [
        ("/health", True),
        ("/login", True),
        ("/setup", True),
        ("/static/css/app.css", True),
        ("/", False),
        ("/settings", False),
        ("/api/stream/start", False),
        ("/api/visuals/blocks", False),
    ],
)
def test_public_paths(path: str, public: bool):
    assert is_public(path) is public


def test_beat_alignment_is_reachable_from_the_dashboard():
    """It is the switch for the whole tempo feature — the BPM column, the
    "Analyse tempo" button and the per-track crossfade length all do nothing
    without it. Left out of this table it could only be set with an
    environment variable, which for a container user means editing the compose
    file and recreating the stack to try a setting.
    """
    from app.services.system.settings import EDITABLE

    assert "audio.crossfade.beat_aligned" in EDITABLE
    assert "audio.crossfade.beat_align_cue_in" in EDITABLE


def test_every_editable_path_exists_in_the_configuration():
    """A typo here is invisible: the dashboard shows a row that writes nothing."""
    from app.core.config import load_config
    from app.services.system.settings import EDITABLE

    cfg = load_config(None, environ={})
    for path in EDITABLE:
        cursor = cfg
        for part in path.split("."):
            assert hasattr(cursor, part), f"{path} does not exist in the configuration"
            cursor = getattr(cursor, part)
