from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR
from .calibration_analysis import build_calibration_summary


def write_calibration_sensitivity_outputs() -> dict[str, Path]:
    outputs = build_calibration_summary()
    summary = pd.read_csv(outputs["calibration_summary"])
    rows = []
    for (dataset, target, split), group in summary.groupby(["dataset", "target", "split"], dropna=False):
        best_ece = group.sort_values("ece").iloc[0]
        best_brier = group.sort_values("brier_score").iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "split": split,
                "best_ece_model": best_ece["model"],
                "best_ece": best_ece["ece"],
                "best_brier_model": best_brier["model"],
                "best_brier": best_brier["brier_score"],
                "note": "Calibration sensitivity across calibrated/fusion/neuro-symbolic variants; lower is better.",
            }
        )
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration_sensitivity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"calibration_sensitivity": path, **outputs}
