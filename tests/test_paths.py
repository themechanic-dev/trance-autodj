from __future__ import annotations

from pathlib import Path

from app.core.config import load_config
from app.core.paths import Paths


def test_every_path_sits_under_the_data_dir(tmp_path: Path):
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "d")})
    paths = Paths.from_config(cfg)
    for directory in paths.directories():
        assert paths.root == directory or paths.root in directory.parents


def test_ensure_creates_everything(tmp_path: Path):
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "d")})
    paths = Paths.from_config(cfg)
    paths.ensure()
    for directory in paths.directories():
        assert directory.is_dir()


def test_models_dir_can_live_outside_the_data_dir(tmp_path: Path):
    """Model weights are large; keeping them off a mirrored data volume matters."""
    elsewhere = tmp_path / "ai-models"
    cfg = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "d"),
            "TAD_VISUAL__AI__MODELS_DIR": str(elsewhere),
        },
    )
    paths = Paths.from_config(cfg)
    assert paths.models == elsewhere.resolve()


def test_fifo_name_is_honoured(tmp_path: Path):
    cfg = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "d"),
            "TAD_STREAM__FIFO_NAME": "custom.fifo",
        },
    )
    assert Paths.from_config(cfg).video_fifo.name == "custom.fifo"
