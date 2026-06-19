from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def build_coverage_utility_tradeoff(results: pd.DataFrame | None = None) -> pd.DataFrame:
    if results is None:
        path = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_conformal_results.csv"
        results = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if results.empty:
        return pd.DataFrame()
    rows = []
    for _, row in results.iterrows():
        for burden_penalty in [0.00, 0.02, 0.05, 0.10, 0.20]:
            utility = (1.0 - float(row["harmful_rate_among_non_intervened"])) * float(row["non_intervention_rate"]) - burden_penalty * float(row["intervention_burden"])
            rows.append(
                {
                    "dataset": row["dataset"],
                    "target": row["target"],
                    "split": row["split"],
                    "model": row["model"],
                    "burden_penalty": burden_penalty,
                    "non_intervention_rate": row["non_intervention_rate"],
                    "intervention_burden": row["intervention_burden"],
                    "harmful_rate_among_non_intervened": row["harmful_rate_among_non_intervened"],
                    "utility_proxy": utility,
                }
            )
    return pd.DataFrame(rows)


def write_coverage_utility_tradeoff() -> dict[str, Path]:
    out = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_coverage_utility_tradeoff.csv"
    build_coverage_utility_tradeoff().to_csv(out, index=False)
    return {"reliaguard_coverage_utility_tradeoff": out}

