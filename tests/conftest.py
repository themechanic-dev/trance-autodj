from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Config, load_config
from app.core.paths import Paths


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def config(data_dir: Path) -> Config:
    return load_config(None, environ={"TAD_APP__DATA_DIR": str(data_dir)})


@pytest.fixture
def paths(config: Config) -> Paths:
    p = Paths.from_config(config)
    p.ensure()
    return p
