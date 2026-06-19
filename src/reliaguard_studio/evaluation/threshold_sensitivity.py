from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def write_threshold_sensitivity_outputs() -> dict[str, Path]:
    predictions_path = REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv"
    if not predictions_path.exists():
        return {}
    frame = pd.read_csv(predictions_path)
    score_col = "y_score" if "y_score" in frame.columns else "y_prob" if "y_prob" in frame.columns else "score"
    if score_col not in frame.columns or "y_true" not in frame.columns:
        return {}
    rows = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for keys, group in frame.groupby(["dataset", "target", "split", "model"], dropna=False):
            dataset, target, split, model = keys
            scores = pd.to_numeric(group[score_col], errors="coerce")
            truth = pd.to_numeric(group["y_true"], errors="coerce")
            mask = scores >= threshold
            harmful = truth == 1
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "split": split,
                    "model": model,
                    "threshold": threshold,
                    "n": int(len(group)),
                    "flagged_fraction": float(mask.mean()),
                    "harmful_capture": float((mask & harmful).sum() / max(harmful.sum(), 1)),
                    "harmful_rate_unflagged": float((harmful & ~mask).sum() / max((~mask).sum(), 1)),
                    "note": "Score-threshold sensitivity; not conformal risk control.",
                }
            )
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "threshold_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"threshold_sensitivity": out}
