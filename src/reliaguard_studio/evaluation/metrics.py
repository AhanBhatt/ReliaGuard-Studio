from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
    }
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        accuracy = np.mean(y_true[mask])
        confidence = np.mean(y_prob[mask])
        ece += np.abs(accuracy - confidence) * np.sum(mask) / total
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if np.any(mask):
            rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": int(np.sum(mask)),
                    "mean_confidence": float(np.mean(y_prob[mask])),
                    "accuracy": float(np.mean(y_true[mask])),
                }
            )
    return rows


def bootstrap_confidence_interval(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            estimates.append(metric_fn(y_true[indices], y_pred[indices]))
        except Exception:
            continue
    if not estimates:
        return float("nan"), float("nan")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def paired_bootstrap_difference(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    first_pred: np.ndarray,
    second_pred: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            first = metric_fn(y_true[indices], first_pred[indices])
            second = metric_fn(y_true[indices], second_pred[indices])
            diffs.append(second - first)
        except Exception:
            continue
    if not diffs:
        return {"mean_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean_difference": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
