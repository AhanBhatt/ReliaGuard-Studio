from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def build_conformal_ablation(results: pd.DataFrame | None = None) -> pd.DataFrame:
    if results is None:
        path = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_conformal_results.csv"
        results = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if results.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, target, split), group in results.groupby(["dataset", "target", "split"], dropna=False):
        best = group.sort_values(["harmful_rate_among_non_intervened", "intervention_burden"]).iloc[0]
        for _, row in group.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "split": split,
                    "model": row["model"],
                    "harmful_rate_delta_vs_best": float(row["harmful_rate_among_non_intervened"] - best["harmful_rate_among_non_intervened"]),
                    "burden_delta_vs_best": float(row["intervention_burden"] - best["intervention_burden"]),
                    "utility_delta_vs_best": float(row["expected_utility_proxy"] - best["expected_utility_proxy"]),
                    "best_reference_model": best["model"],
                }
            )
    return pd.DataFrame(rows)


def write_conformal_ablation() -> dict[str, Path]:
    out = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_conformal_ablation.csv"
    build_conformal_ablation().to_csv(out, index=False)
    return {"reliaguard_conformal_ablation": out}

