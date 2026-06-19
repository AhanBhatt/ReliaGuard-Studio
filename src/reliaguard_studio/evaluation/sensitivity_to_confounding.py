from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def write_confounding_sensitivity_grid() -> dict[str, Path]:
    policy_bounds = REAL_DATA_EXPERIMENTS_DIR / "policy_value_bounds.csv"
    if policy_bounds.exists():
        policy = pd.read_csv(policy_bounds)
    else:
        policy = pd.DataFrame()
    rows = []
    if policy.empty:
        rows.append({"status": "not_available", "reason": "policy_value_bounds.csv missing"})
    else:
        for _, row in policy.iterrows():
            for gamma in [0.00, 0.02, 0.05, 0.10, 0.20]:
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "policy": row["policy"],
                        "unmeasured_confounding_penalty": gamma,
                        "sensitivity_adjusted_gain": float(row["observational_utility_gain"]) - gamma,
                        "interpretation": "Policy prioritization remains positive only if adjusted gain stays above zero; not a causal proof.",
                    }
                )
    path = REAL_DATA_EXPERIMENTS_DIR / "confounding_sensitivity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"confounding_sensitivity": path}

