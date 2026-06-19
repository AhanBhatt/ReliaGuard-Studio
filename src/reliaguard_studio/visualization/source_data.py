from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import PAPER_SOURCE_DATA_DIR


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.lower()).strip("_")


def write_source_data(figure_stem: str, panel: str, data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> Path:
    """Write the plotted data behind one figure panel."""

    PAPER_SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(f"{figure_stem}_{panel}")
    if isinstance(data, pd.DataFrame):
        path = PAPER_SOURCE_DATA_DIR / f"{stem}.csv"
        data.to_csv(path, index=False)
        return path
    path = PAPER_SOURCE_DATA_DIR / f"{stem}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def source_data_paths(figure_stem: str) -> list[Path]:
    PAPER_SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _safe_name(figure_stem)
    return sorted(PAPER_SOURCE_DATA_DIR.glob(f"{prefix}_*.*"))
