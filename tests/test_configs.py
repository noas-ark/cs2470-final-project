"""Every committed experiment config must parse through the runner's own loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from s1 import config

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.toml"))


def test_configs_exist() -> None:
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path: Path) -> None:
    assert config.load(path)
