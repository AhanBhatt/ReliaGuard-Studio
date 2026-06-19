from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def write_harm_weight_sensitivity_outputs() -> dict[str, Path]:
    policy_path = REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv"
    if not policy_path.exists():
        return {}
    frame = pd.read_csv(policy_path)
    rows = []
    for over_weight in [1.0, 2.0, 3.0, 5.0]:
        for under_weight in [0.5, 1.0, 1.5, 2.0]:
            for _, row in frame.iterrows():
                utility = (
                    float(row["expected_final_correct"])
                    - over_weight * float(row["expected_overreliance"])
                    - under_weight * float(row["expected_underreliance"])
                    - 0.05 * float(row["intervention_burden"])
                )
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "policy": row["policy"],
                        "overreliance_weight": over_weight,
                        "underreliance_weight": under_weight,
                        "intervention_burden_weight": 0.05,
                        "sensitivity_utility": utility,
                        "note": "Observational utility sensitivity; not causal intervention evidence.",
                    }
                )
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "harm_weight_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"harm_weight_sensitivity": out}
