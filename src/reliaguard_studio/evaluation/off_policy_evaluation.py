from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR


def _load_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for dataset_dir in REAL_DATA_PREPARED_DIR.iterdir():
        path = dataset_dir / "interactions.csv"
        if path.exists():
            frames[dataset_dir.name] = pd.read_csv(path, low_memory=False)
    return frames


def classify_policy_evidence(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    frames = frames or _load_frames()
    rows = []
    for dataset, frame in frames.items():
        has_condition = "condition_id" in frame.columns or "condition_name" in frame.columns
        has_reliance = {"overreliance", "underreliance", "final_correct"}.issubset(frame.columns)
        if dataset in {"chi2023_dke", "convxai_iui2025"} and has_condition and has_reliance:
            status = "randomized or condition-structured public experiment"
            method = "condition contrasts, participant-cluster GEE, conservative policy prioritization"
            allowed = "Associations between interface condition and reliance; prioritization of gating candidates."
            disallowed = "Causal effect of a new ReliaGuard-NS gating intervention."
        elif dataset == "haiid" and has_reliance:
            status = "advice-source/task randomized or structured, no gating intervention"
            method = "clustered contrasts, conformal risk control on held-out predictions"
            allowed = "Reliance-state detection and risk-control diagnostics."
            disallowed = "Causal effect of verification prompts or withholding advice."
        elif dataset in {"pardos_chatgpt_tutoring", "flora_ips"}:
            status = "learning/process evidence, not reliance-policy eligible"
            method = "learning/process analysis only"
            allowed = "Learning-gain or process-trace associations within dataset limits."
            disallowed = "Reliance-gating policy claims."
        else:
            status = "prediction-only or not eligible"
            method = "not used for policy evaluation"
            allowed = "None for policy conclusions."
            disallowed = "Policy value or causal claims."
        rows.append(
            {
                "dataset": dataset,
                "policy_evidence_class": status,
                "policy_evaluation_method": method,
                "causal_status": "non-causal for ReliaGuard-NS intervention",
                "conclusion_allowed": allowed,
                "conclusion_not_allowed": disallowed,
            }
        )
    return pd.DataFrame(rows)


def run_off_policy_evaluation(frames: dict[str, pd.DataFrame] | None = None) -> dict[str, Path]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    evidence = classify_policy_evidence(frames)
    evidence_path = REAL_DATA_EXPERIMENTS_DIR / "policy_evidence_boundary.csv"
    evidence.to_csv(evidence_path, index=False)

    policy_path = REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv"
    policy = pd.read_csv(policy_path) if policy_path.exists() else pd.DataFrame()
    rows = []
    if not policy.empty:
        for dataset, group in policy.groupby("dataset", dropna=False):
            baseline = group.loc[group["policy"] == "observed_no_gating"]
            base_utility = float(baseline["expected_utility"].iloc[0]) if not baseline.empty else float(group["expected_utility"].median())
            for _, row in group.iterrows():
                utility_gain = float(row["expected_utility"] - base_utility)
                uncertainty_width = float(row.get("utility_ci_high", row["expected_utility"]) - row.get("utility_ci_low", row["expected_utility"]))
                rows.append(
                    {
                        "dataset": dataset,
                        "policy": row["policy"],
                        "observational_utility_gain": utility_gain,
                        "lower_bound_conservative_gain": utility_gain - 0.5 * uncertainty_width,
                        "upper_bound_conservative_gain": utility_gain + 0.5 * uncertainty_width,
                        "evidence_status": "observational utility simulation; prospective validation required",
                    }
                )
    bounds = pd.DataFrame(rows)
    bounds_path = REAL_DATA_EXPERIMENTS_DIR / "policy_value_bounds.csv"
    bounds.to_csv(bounds_path, index=False)
    return {"policy_evidence_boundary": evidence_path, "policy_value_bounds": bounds_path}


def estimate_inverse_propensity_weights(frame: pd.DataFrame, condition_col: str = "condition_name") -> pd.Series:
    if condition_col not in frame.columns:
        return pd.Series(np.ones(len(frame)), index=frame.index)
    probs = frame[condition_col].astype(str).map(frame[condition_col].astype(str).value_counts(normalize=True))
    return (1.0 / probs.replace(0, np.nan)).fillna(1.0).clip(upper=20.0)

