from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def _bootstrap_mean(values: np.ndarray, seed: int = 42, n_boot: int = 200) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    return float(np.mean(values)), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def build_label_definition_sensitivity(interactions_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in ["haiid", "convxai_iui2025", "chi2023_dke"]:
        frame = interactions_map.get(dataset)
        if frame is None:
            continue
        definitions = {
            "conservative_record_rate": {
                "overreliance": frame["overreliance"].to_numpy(dtype=float),
                "underreliance": frame["underreliance"].to_numpy(dtype=float),
            },
            "applicable_case_rate": {
                "overreliance": frame.loc[(frame["initial_correct"] == 1) & (frame["advice_correct"] == 0), "overreliance"].to_numpy(dtype=float),
                "underreliance": frame.loc[(frame["initial_correct"] == 0) & (frame["advice_correct"] == 1), "underreliance"].to_numpy(dtype=float),
            },
            "adoption_only_rate": {
                "overreliance": frame.loc[(frame["initial_correct"] == 1) & (frame["advice_correct"] == 0), "advice_uptake"].to_numpy(dtype=float),
                "underreliance": 1.0 - frame.loc[(frame["initial_correct"] == 0) & (frame["advice_correct"] == 1), "advice_uptake"].to_numpy(dtype=float),
            },
        }
        for definition, outcomes in definitions.items():
            for outcome, values in outcomes.items():
                mean, low, high = _bootstrap_mean(values, seed=42 + len(rows))
                rows.append(
                    {
                        "dataset": dataset,
                        "outcome": outcome,
                        "definition": definition,
                        "n": int(np.sum(~np.isnan(values))),
                        "rate": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "note": "Definitions are sensitivity checks; headline manuscript uses conservative behavioural definitions.",
                    }
                )
    return pd.DataFrame(rows)


def build_decision_curve_summary(policy_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, subframe in policy_frame.groupby("dataset"):
        observed = subframe.loc[subframe["policy"] == "observed_no_gating"]
        if observed.empty:
            continue
        observed_row = observed.iloc[0]
        observed_harm = float(observed_row["expected_overreliance"] + observed_row["expected_underreliance"])
        observed_utility = float(observed_row["expected_utility"])
        for _, row in subframe.iterrows():
            harmful = float(row["expected_overreliance"] + row["expected_underreliance"])
            rows.append(
                {
                    "dataset": dataset,
                    "policy": row["policy"],
                    "intervention_burden": float(row["intervention_burden"]),
                    "expected_utility": float(row["expected_utility"]),
                    "utility_gain_vs_observed": float(row["expected_utility"] - observed_utility),
                    "harmful_reliance_reduction": float(observed_harm - harmful),
                    "final_correct_change": float(row["expected_final_correct"] - observed_row["expected_final_correct"]),
                    "note": "Conservative observational utility frontier, not a causal decision curve.",
                }
            )
    return pd.DataFrame(rows)


def build_missingness_sensitivity(interactions_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    feature_groups = {
        "confidence_features": ["initial_confidence", "final_confidence", "confidence_change", "post_explain_reliability"],
        "condition_features": ["condition_id", "condition_name", "advice_source_label"],
        "process_features": ["prompt_count", "revision_depth", "source_engagement_proxy", "copy_paste_write_ratio"],
    }
    for dataset, frame in interactions_map.items():
        for group, columns in feature_groups.items():
            present = [column for column in columns if column in frame.columns]
            if not present:
                continue
            missing_rate = float(frame[present].isna().mean().mean())
            rows.append(
                {
                    "dataset": dataset,
                    "feature_group": group,
                    "available_columns": "; ".join(present),
                    "mean_missing_rate": missing_rate,
                    "n_records": int(len(frame)),
                    "note": "Missingness audit for leakage and robustness interpretation.",
                }
            )
    return pd.DataFrame(rows)


def write_sensitivity_outputs(interactions_map: dict[str, pd.DataFrame], policy_frame: pd.DataFrame | None = None) -> dict[str, str]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    label_df = build_label_definition_sensitivity(interactions_map)
    missing_df = build_missingness_sensitivity(interactions_map)
    if policy_frame is None:
        policy_path = REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv"
        policy_frame = pd.read_csv(policy_path) if policy_path.exists() else pd.DataFrame()
    decision_curve = build_decision_curve_summary(policy_frame) if not policy_frame.empty else pd.DataFrame()
    sensitivity_summary = pd.concat(
        [
            label_df.assign(sensitivity_type="label_definition"),
            missing_df.rename(columns={"mean_missing_rate": "rate"}).assign(
                outcome="missingness",
                definition=missing_df["feature_group"] if not missing_df.empty else "",
                ci_low=np.nan,
                ci_high=np.nan,
                sensitivity_type="missingness",
            )[["dataset", "outcome", "definition", "n_records", "rate", "ci_low", "ci_high", "note", "sensitivity_type"]].rename(columns={"n_records": "n"}),
        ],
        ignore_index=True,
        sort=False,
    )
    outputs = {
        "label_definition_sensitivity": REAL_DATA_EXPERIMENTS_DIR / "label_definition_sensitivity.csv",
        "decision_curve_summary": REAL_DATA_EXPERIMENTS_DIR / "decision_curve_summary.csv",
        "sensitivity_summary": REAL_DATA_EXPERIMENTS_DIR / "sensitivity_summary.csv",
    }
    label_df.to_csv(outputs["label_definition_sensitivity"], index=False)
    decision_curve.to_csv(outputs["decision_curve_summary"], index=False)
    sensitivity_summary.to_csv(outputs["sensitivity_summary"], index=False)
    return {key: str(path) for key, path in outputs.items()}
