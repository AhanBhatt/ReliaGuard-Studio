from __future__ import annotations

import numpy as np
import pandas as pd


def selective_prediction_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    uncertainty: np.ndarray | None = None,
    coverages: list[float] | None = None,
) -> pd.DataFrame:
    """Return accuracy/error at multiple retained-coverage levels.

    Lower uncertainty is retained first. If uncertainty is unavailable, distance
    from 0.5 is used as a confidence proxy. This is a diagnostic selective
    prediction curve, not a deployment guarantee.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if uncertainty is None:
        uncertainty = 1.0 - np.abs(y_score - 0.5) * 2.0
    uncertainty = np.asarray(uncertainty).astype(float)
    coverages = coverages or [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    order = np.argsort(uncertainty)
    rows = []
    for coverage in coverages:
        n_keep = max(1, int(round(len(order) * coverage)))
        keep = order[:n_keep]
        pred = (y_score[keep] >= 0.5).astype(int)
        accuracy = float(np.mean(pred == y_true[keep]))
        rows.append(
            {
                "coverage": float(coverage),
                "n_retained": int(n_keep),
                "accuracy": accuracy,
                "error_rate": float(1.0 - accuracy),
                "mean_uncertainty_retained": float(np.mean(uncertainty[keep])),
            }
        )
    return pd.DataFrame(rows)

