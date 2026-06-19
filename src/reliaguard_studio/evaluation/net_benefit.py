from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def write_net_benefit_table() -> dict[str, Path]:
    policy_path = REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv"
    policy = pd.read_csv(policy_path) if policy_path.exists() else pd.DataFrame()
    rows = []
    for _, row in policy.iterrows():
        for harm_cost in [1.0, 2.0, 4.0]:
            net = float(row["expected_final_correct"]) - harm_cost * float(row["expected_overreliance"] + row["expected_underreliance"]) - 0.05 * float(row["intervention_burden"])
            rows.append(
                {
                    "dataset": row["dataset"],
                    "policy": row["policy"],
                    "harm_cost_ratio": harm_cost,
                    "net_benefit_proxy": net,
                    "note": "Decision-curve style observational net-benefit proxy; not a causal treatment effect.",
                }
            )
    path = REAL_DATA_EXPERIMENTS_DIR / "net_benefit.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"net_benefit": path}

