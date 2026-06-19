from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR
from .metrics import classification_metrics


def _calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    try:
        import statsmodels.api as sm

        clipped = np.clip(y_prob, 1e-5, 1 - 1e-5)
        logit = np.log(clipped / (1 - clipped))
        x = sm.add_constant(logit)
        result = sm.Logit(y_true, x).fit(disp=False, maxiter=80)
        return float(result.params[1]), float(result.params[0])
    except Exception:
        return float("nan"), float("nan")


def _bin_curve(frame: pd.DataFrame, mode: str, n_bins: int = 10) -> pd.DataFrame:
    data = frame.copy()
    if mode == "adaptive":
        try:
            data["bin"] = pd.qcut(data["y_prob"], q=min(n_bins, data["y_prob"].nunique()), duplicates="drop")
        except ValueError:
            data["bin"] = pd.cut(data["y_prob"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    else:
        data["bin"] = pd.cut(data["y_prob"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    rows = []
    for interval, subframe in data.groupby("bin", observed=True):
        if subframe.empty:
            continue
        rate = float(subframe["y_true"].mean())
        n = len(subframe)
        se = float(np.sqrt(max(rate * (1 - rate), 1e-9) / n))
        rows.append(
            {
                "binning": mode,
                "bin": str(interval),
                "n": int(n),
                "mean_confidence": float(subframe["y_prob"].mean()),
                "accuracy": rate,
                "ci_low": max(0.0, rate - 1.96 * se),
                "ci_high": min(1.0, rate + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


def build_calibration_summary(predictions: pd.DataFrame | None = None) -> dict[str, Path]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if predictions is None:
        predictions = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv")
    summary_rows = []
    curve_rows = []
    for keys, subframe in predictions.groupby(["dataset", "target", "split", "model"]):
        dataset, target, split, model = keys
        y_true = subframe["y_true"].to_numpy(dtype=int)
        y_prob = subframe["y_prob"].to_numpy(dtype=float)
        metrics = classification_metrics(y_true, y_prob)
        slope, intercept = _calibration_slope(y_true, y_prob)
        bins = _bin_curve(subframe, "fixed")
        mce = float((bins["accuracy"] - bins["mean_confidence"]).abs().max()) if not bins.empty else float("nan")
        summary_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "split": split,
                "model": model,
                "n": int(len(subframe)),
                "auroc": metrics["auroc"],
                "brier_score": metrics["brier_score"],
                "ece": metrics["ece"],
                "mce": mce,
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
        for mode in ["fixed", "adaptive"]:
            curve = _bin_curve(subframe, mode)
            for row in curve.to_dict("records"):
                curve_rows.append({"dataset": dataset, "target": target, "split": split, "model": model, **row})
    summary = pd.DataFrame(summary_rows)
    curves = pd.DataFrame(curve_rows)
    summary_path = REAL_DATA_EXPERIMENTS_DIR / "calibration_summary.csv"
    curves_path = REAL_DATA_EXPERIMENTS_DIR / "calibration_curves_detailed.csv"
    summary.to_csv(summary_path, index=False)
    curves.to_csv(curves_path, index=False)
    return {"calibration_summary": summary_path, "calibration_curves_detailed": curves_path}
