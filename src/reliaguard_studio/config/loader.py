from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from .schemas import ProjectConfig


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_project_config(path: Path | None = None) -> ProjectConfig:
    if path is None:
        with resources.files("reliaguard_studio.config").joinpath("default_config.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            raw = yaml.safe_load(handle)
    else:
        raw = _read_yaml(path)
    return ProjectConfig.model_validate(raw)
