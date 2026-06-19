from __future__ import annotations

import numpy as np

from reliaguard_studio.evaluation.metrics import (
    bootstrap_confidence_interval,
    classification_metrics,
    paired_bootstrap_difference,
    regression_metrics,
    reliability_curve,
)


def test_classification_metrics_and_reliability_curve() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.10, 0.20, 0.75, 0.90, 0.65, 0.35, 0.80, 0.15])
    metrics = classification_metrics(y_true, y_prob)
    curve = reliability_curve(y_true, y_prob, n_bins=5)

    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert sum(row["count"] for row in curve) == len(y_true)


def test_regression_metrics_and_bootstrap_helpers() -> None:
    y_true = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_pred = np.array([0.12, 0.28, 0.49, 0.74, 0.85])
    metrics = regression_metrics(y_true, y_pred)
    ci = bootstrap_confidence_interval(lambda a, b: float(np.mean(np.abs(a - b))), y_true, y_pred, n_boot=20)
    diff = paired_bootstrap_difference(
        lambda a, b: float(np.mean(np.abs(a - b))),
        y_true,
        y_pred,
        np.array([0.11, 0.31, 0.51, 0.71, 0.88]),
        n_boot=20,
    )

    assert metrics["mae"] < 0.1
    assert len(ci) == 2
    assert set(diff) == {"mean_difference", "ci_low", "ci_high"}
