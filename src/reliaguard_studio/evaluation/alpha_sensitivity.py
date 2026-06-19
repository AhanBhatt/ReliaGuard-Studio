from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..models.reliaguard_ns import evaluate_reliaguard_predictions
from ..paths import PAPER_SOURCE_DATA_DIR, REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR


DEFAULT_ALPHAS = [0.01, 0.05, 0.10, 0.20, 0.30]


def _prepared_final_correct() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    dataset_dirs = REAL_DATA_PREPARED_DIR.iterdir() if REAL_DATA_PREPARED_DIR.exists() else []
    for dataset_dir in dataset_dirs:
        path = dataset_dir / "interactions.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        keep = [col for col in ["participant_id", "task_instance_key", "final_correct"] if col in frame.columns]
        if len(keep) == 3:
            subset = frame[keep].copy()
            subset["dataset"] = dataset_dir.name
            rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=["dataset", "participant_id", "task_instance_key", "final_correct"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["dataset", "participant_id", "task_instance_key"])


def _with_allowed_final_correct(results: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    if results.empty or actions.empty:
        return results
    correctness = _prepared_final_correct()
    if correctness.empty:
        results["final_correct_among_non_intervened"] = pd.NA
        return results
    merged = actions.merge(correctness, on=["dataset", "participant_id", "task_instance_key"], how="left")
    allowed = merged.loc[merged["reliaguard_action"].eq("no_intervention")].copy()
    if allowed.empty:
        results["final_correct_among_non_intervened"] = pd.NA
        return results
    grouped = (
        allowed.groupby(["dataset", "target", "split", "model", "alpha"], dropna=False)["final_correct"]
        .mean()
        .reset_index()
        .rename(columns={"final_correct": "final_correct_among_non_intervened"})
    )
    return results.merge(grouped, on=["dataset", "target", "split", "model", "alpha"], how="left")


def build_alpha_sensitivity(predictions: pd.DataFrame | None = None, alphas: list[float] | None = None) -> pd.DataFrame:
    """Evaluate ReliaGuard-NS thresholds over a grid of target alpha values."""
    if predictions is None:
        predictions = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv", low_memory=False)
    rows: list[pd.DataFrame] = []
    for alpha in alphas or DEFAULT_ALPHAS:
        results, actions = evaluate_reliaguard_predictions(predictions, alpha=alpha)
        if results.empty:
            continue
        enriched = _with_allowed_final_correct(results, actions)
        enriched["sensitivity_type"] = "alpha_sensitivity"
        enriched["comparison_family"] = enriched["model"].map(
            {
                "calibrated_gradient_boosting": "tabular-only gating",
                "calibrated_logistic_regression": "tabular-only gating",
                "gradient_boosting": "tabular-only gating",
                "logistic_regression": "tabular-only gating",
                "symbolic_only": "symbolic-only gating",
                "uncertainty_aware_fusion": "uncertainty-aware fusion gating",
                "reliance_state_neurosymbolic": "ReliaGuard-NS",
                "weighted_fusion": "weighted fusion gating",
                "learned_fusion": "learned fusion gating",
            }
        ).fillna("model-based gating")
        rows.append(enriched)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def write_alpha_sensitivity_outputs(alphas: list[float] | None = None) -> dict[str, Path]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    sensitivity_dir = REAL_DATA_EXPERIMENTS_DIR / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    frame = build_alpha_sensitivity(alphas=alphas)
    sensitivity_path = sensitivity_dir / "alpha_sensitivity.csv"
    selective_path = REAL_DATA_EXPERIMENTS_DIR / "conformal_selective_risk_results.csv"
    source_path = PAPER_SOURCE_DATA_DIR / "figure_reliaguard_selective_risk.csv"
    PAPER_SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(sensitivity_path, index=False)
    frame.to_csv(selective_path, index=False)
    frame.to_csv(source_path, index=False)
    return {
        "alpha_sensitivity": sensitivity_path,
        "conformal_selective_risk_results": selective_path,
        "figure_reliaguard_selective_risk_source": source_path,
    }
