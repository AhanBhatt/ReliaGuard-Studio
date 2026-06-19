from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR
from .sensitivity import build_label_definition_sensitivity


def write_label_definition_sensitivity_outputs() -> dict[str, Path]:
    interactions: dict[str, pd.DataFrame] = {}
    dataset_dirs = REAL_DATA_PREPARED_DIR.iterdir() if REAL_DATA_PREPARED_DIR.exists() else []
    for dataset_dir in dataset_dirs:
        path = dataset_dir / "interactions.csv"
        if path.exists():
            interactions[dataset_dir.name] = pd.read_csv(path, low_memory=False)
    frame = build_label_definition_sensitivity(interactions)
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "label_definition_sensitivity.csv"
    root_path = REAL_DATA_EXPERIMENTS_DIR / "label_definition_sensitivity.csv"
    frame.to_csv(path, index=False)
    frame.to_csv(root_path, index=False)
    return {"label_definition_sensitivity": path, "label_definition_sensitivity_root": root_path}
