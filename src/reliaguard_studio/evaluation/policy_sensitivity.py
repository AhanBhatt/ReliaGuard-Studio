from __future__ import annotations

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def build_policy_burden_sensitivity(policy_frame: pd.DataFrame, kappas: list[float] | None = None) -> pd.DataFrame:
    kappas = kappas or [0.00, 0.02, 0.05, 0.10, 0.15]
    rows = []
    if policy_frame.empty:
        return pd.DataFrame()
    for dataset, sub in policy_frame.groupby("dataset"):
        observed = sub.loc[sub["policy"].eq("observed_no_gating")]
        if observed.empty:
            continue
        for kappa in kappas:
            observed_utility = float(
                observed["expected_final_correct"].iloc[0]
                - 2.0 * observed["expected_overreliance"].iloc[0]
                - observed["expected_underreliance"].iloc[0]
                - kappa * observed["intervention_burden"].iloc[0]
            )
            for _, row in sub.iterrows():
                utility = float(
                    row["expected_final_correct"]
                    - 2.0 * row["expected_overreliance"]
                    - row["expected_underreliance"]
                    - kappa * row["intervention_burden"]
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "policy": row["policy"],
                        "burden_penalty_kappa": kappa,
                        "expected_utility": utility,
                        "utility_gain_vs_observed": utility - observed_utility,
                        "note": "Sensitivity of observational policy utility to intervention-burden penalty.",
                    }
                )
    return pd.DataFrame(rows)


def write_policy_sensitivity_outputs(policy_frame: pd.DataFrame | None = None) -> dict[str, str]:
    if policy_frame is None:
        path = REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv"
        policy_frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    sensitivity = build_policy_burden_sensitivity(policy_frame)
    path = out_dir / "policy_burden_sensitivity.csv"
    root_path = REAL_DATA_EXPERIMENTS_DIR / "policy_burden_sensitivity.csv"
    sensitivity.to_csv(path, index=False)
    sensitivity.to_csv(root_path, index=False)
    return {"policy_burden_sensitivity": str(path), "policy_burden_sensitivity_root": str(root_path)}

