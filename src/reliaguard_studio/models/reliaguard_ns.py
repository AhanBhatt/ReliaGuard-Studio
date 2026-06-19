from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation.conformal_risk_control import (
    assign_conformal_actions,
    deterministic_split_key,
    split_conformal_threshold,
    summarize_selective_risk,
)


@dataclass(frozen=True)
class ReliaGuardConfig:
    alpha: float = 0.10
    calibration_fraction: float = 0.50
    minimum_positive_calibration: int = 8


class ReliaGuardNS:
    """ReliaGuard-NS conformal reliance control.

    ReliaGuard-NS wraps a harmful-reliance risk score with a split-conformal
    threshold. The implementation is intentionally modest: the finite-sample
    statement applies under exchangeability of calibration and deployment
    examples for the evaluated dataset/target/model group. It is not a causal
    claim about the effect of showing an intervention.
    """

    def __init__(self, config: ReliaGuardConfig | None = None) -> None:
        self.config = config or ReliaGuardConfig()

    @staticmethod
    def _risk_score(frame: pd.DataFrame) -> pd.Series:
        base = frame["y_prob"].astype(float).clip(0.0, 1.0)
        if "uncertainty_proxy" in frame.columns:
            uncertainty = frame["uncertainty_proxy"].astype(float).fillna(0.0).clip(0.0, 1.0)
            return (0.90 * base + 0.10 * uncertainty).clip(0.0, 1.0)
        return base

    def evaluate_group(self, frame: pd.DataFrame) -> tuple[dict[str, float | int | str], pd.DataFrame] | None:
        if frame.empty:
            return None
        dataset = str(frame["dataset"].iloc[0])
        target = str(frame["target"].iloc[0])
        split = str(frame["split"].iloc[0])
        model = str(frame["model"].iloc[0])
        working = frame.copy()
        working["y_true"] = working["y_true"].astype(int)
        working["risk_score"] = self._risk_score(working)
        working["calibration_key"] = [
            deterministic_split_key(dataset, target, split, model, row.participant_id, row.task_instance_key, idx)
            for idx, row in enumerate(working.itertuples(index=False))
        ]
        calibration = working.loc[working["calibration_key"] < self.config.calibration_fraction].copy()
        test = working.loc[working["calibration_key"] >= self.config.calibration_fraction].copy()
        if calibration.empty or test.empty:
            return None
        threshold = split_conformal_threshold(
            calibration["risk_score"],
            calibration["y_true"],
            alpha=self.config.alpha,
            minimum_positive=self.config.minimum_positive_calibration,
        )
        if threshold is None:
            return None
        summary = summarize_selective_risk(test["risk_score"], test["y_true"], threshold.threshold)
        summary.update(
            {
                "dataset": dataset,
                "target": target,
                "split": split,
                "model": model,
                "method": "ReliaGuard-NS",
                "alpha": self.config.alpha,
                "n_calibration": threshold.n_calibration,
                "n_positive_calibration": threshold.n_positive_calibration,
                "finite_sample_slack": threshold.finite_sample_slack,
                "missed_harmful_bound": min(1.0, self.config.alpha + threshold.finite_sample_slack),
                "guarantee_scope": "Split-conformal harmful-case recall under exchangeability; not causal intervention evidence.",
            }
        )
        actions = assign_conformal_actions(test, threshold.threshold, target)
        actions["alpha"] = self.config.alpha
        actions["finite_sample_slack"] = threshold.finite_sample_slack
        actions["missed_harmful_bound"] = min(1.0, self.config.alpha + threshold.finite_sample_slack)
        return summary, actions


def evaluate_reliaguard_predictions(predictions: pd.DataFrame, alpha: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"dataset", "target", "split", "model", "participant_id", "task_instance_key", "y_true", "y_prob"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")
    allowed_datasets = {"haiid", "chi2023_dke", "convxai_iui2025"}
    allowed_targets = {"overreliance", "underreliance", "appropriate_reliance"}
    allowed_models = {
        "calibrated_gradient_boosting",
        "calibrated_logistic_regression",
        "gradient_boosting",
        "learned_fusion",
        "logistic_regression",
        "reliance_state_neurosymbolic",
        "symbolic_only",
        "uncertainty_aware_fusion",
        "weighted_fusion",
    }
    working = predictions.loc[
        predictions["dataset"].isin(allowed_datasets)
        & predictions["target"].isin(allowed_targets)
        & predictions["model"].isin(allowed_models)
    ].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()
    working["y_true"] = pd.to_numeric(working["y_true"], errors="coerce").fillna(0).astype(int)
    working["y_prob"] = pd.to_numeric(working["y_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    # For appropriate reliance, control the complement: harmful or inappropriate reliance.
    mask = working["target"] == "appropriate_reliance"
    working.loc[mask, "y_true"] = 1 - working.loc[mask, "y_true"]
    working.loc[mask, "y_prob"] = 1.0 - working.loc[mask, "y_prob"]
    working.loc[mask, "target"] = "inappropriate_reliance"

    model = ReliaGuardNS(ReliaGuardConfig(alpha=alpha))
    summaries: list[dict[str, float | int | str]] = []
    actions: list[pd.DataFrame] = []
    group_cols = ["dataset", "target", "split", "model"]
    for _, group in working.groupby(group_cols, dropna=False):
        result = model.evaluate_group(group)
        if result is None:
            continue
        summary, group_actions = result
        summaries.append(summary)
        actions.append(group_actions)
    summary_df = pd.DataFrame(summaries)
    action_df = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame()
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["dataset", "target", "split", "model"]).reset_index(drop=True)
    if not action_df.empty:
        action_df = action_df.sort_values(["dataset", "target", "split", "model", "participant_id"]).reset_index(drop=True)
    return summary_df, action_df


def summarize_reliaguard_by_dataset(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    numeric = [
        "harmful_rate_among_non_intervened",
        "missed_harmful_fraction",
        "empirical_harmful_capture",
        "intervention_burden",
        "non_intervention_rate",
        "expected_utility_proxy",
    ]
    grouped = results.groupby(["dataset", "target"], dropna=False)[numeric].median().reset_index()
    grouped["summary_note"] = "Median across model/split conformal evaluations; lower harmful rate and burden are better, conditional on target."
    return grouped
