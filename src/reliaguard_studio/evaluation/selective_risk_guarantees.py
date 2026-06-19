from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..models.reliaguard_ns import evaluate_reliaguard_predictions, summarize_reliaguard_by_dataset
from ..paths import REAL_DATA_EXPERIMENTS_DIR


def run_conformal_risk_control(alpha: float = 0.10) -> dict[str, Path]:
    """Run ReliaGuard-NS on held-out prediction artifacts."""
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError("real_predictions.csv is missing; run `nsca run-real-experiments` first.")
    predictions = pd.read_csv(predictions_path, low_memory=False)
    results, actions = evaluate_reliaguard_predictions(predictions, alpha=alpha)
    dataset_summary = summarize_reliaguard_by_dataset(results)
    results_path = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_conformal_results.csv"
    actions_path = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_conformal_actions.csv"
    summary_path = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_dataset_summary.csv"
    results.to_csv(results_path, index=False)
    actions.to_csv(actions_path, index=False)
    dataset_summary.to_csv(summary_path, index=False)
    guarantee_path = REAL_DATA_EXPERIMENTS_DIR / "selective_risk_guarantees.csv"
    if results.empty:
        guarantee = pd.DataFrame(
            [
                {
                    "status": "not_estimated",
                    "reason": "No eligible prediction groups had enough positive calibration examples.",
                }
            ]
        )
    else:
        guarantee = results[
            [
                "dataset",
                "target",
                "split",
                "model",
                "alpha",
                "threshold",
                "n_positive_calibration",
                "finite_sample_slack",
                "missed_harmful_bound",
                "missed_harmful_fraction",
                "harmful_rate_among_non_intervened",
                "intervention_burden",
                "guarantee_scope",
            ]
        ].copy()
    guarantee.to_csv(guarantee_path, index=False)
    return {
        "reliaguard_conformal_results": results_path,
        "reliaguard_conformal_actions": actions_path,
        "reliaguard_dataset_summary": summary_path,
        "selective_risk_guarantees": guarantee_path,
    }

