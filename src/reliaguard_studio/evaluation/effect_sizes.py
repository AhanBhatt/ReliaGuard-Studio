from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import bootstrap_confidence_interval


def mean_difference_with_ci(values: pd.Series, seed: int = 42, n_boot: int = 200) -> dict[str, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return {"effect": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    low, high = bootstrap_confidence_interval(lambda a, b: float(np.mean(a)), array, array, n_boot=n_boot, seed=seed)
    return {"effect": float(np.mean(array)), "ci_low": low, "ci_high": high, "n": int(len(array))}


def standardized_mean_difference(left: pd.Series, right: pd.Series) -> float:
    left_values = left.dropna().to_numpy(dtype=float)
    right_values = right.dropna().to_numpy(dtype=float)
    if len(left_values) < 2 or len(right_values) < 2:
        return float("nan")
    pooled = np.sqrt(((len(left_values) - 1) * np.var(left_values, ddof=1) + (len(right_values) - 1) * np.var(right_values, ddof=1)) / (len(left_values) + len(right_values) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(left_values) - np.mean(right_values)) / pooled)


def odds_ratio_from_2x2(a: float, b: float, c: float, d: float, smoothing: float = 0.5) -> dict[str, float]:
    """Return a Haldane-Anscombe-smoothed odds ratio and log-scale Wald interval."""
    a_s, b_s, c_s, d_s = a + smoothing, b + smoothing, c + smoothing, d + smoothing
    odds_ratio = (a_s * d_s) / (b_s * c_s)
    log_or = np.log(odds_ratio)
    se = np.sqrt((1 / a_s) + (1 / b_s) + (1 / c_s) + (1 / d_s))
    return {
        "odds_ratio": float(odds_ratio),
        "ci_low": float(np.exp(log_or - 1.96 * se)),
        "ci_high": float(np.exp(log_or + 1.96 * se)),
    }
