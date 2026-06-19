from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REPO_ROOT


def _normal_approx_two_proportion_n(
    baseline_rate: float = 0.25,
    target_rate: float = 0.18,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    # z values for the default preregistered alpha/power. The function accepts
    # parameters but avoids scipy as a hard dependency for this lightweight audit.
    z_alpha = 1.96 if alpha == 0.05 else 1.96
    z_power = 0.84 if power == 0.80 else 0.84
    pooled = (baseline_rate + target_rate) / 2
    delta = abs(baseline_rate - target_rate)
    if delta <= 0:
        return 0
    numerator = z_alpha * sqrt(2 * pooled * (1 - pooled)) + z_power * sqrt(
        baseline_rate * (1 - baseline_rate) + target_rate * (1 - target_rate)
    )
    return int(ceil((numerator / delta) ** 2))


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def write_power_analysis() -> dict[str, Path]:
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "prospective_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [
        {"baseline_harmful_rate": 0.25, "target_harmful_rate": 0.18, "alpha": 0.05, "power": 0.80},
        {"baseline_harmful_rate": 0.30, "target_harmful_rate": 0.21, "alpha": 0.05, "power": 0.80},
        {"baseline_harmful_rate": 0.20, "target_harmful_rate": 0.14, "alpha": 0.05, "power": 0.80},
    ]
    rows = []
    for scenario in scenarios:
        n_per_arm = _normal_approx_two_proportion_n(
            baseline_rate=scenario["baseline_harmful_rate"],
            target_rate=scenario["target_harmful_rate"],
            alpha=scenario["alpha"],
            power=scenario["power"],
        )
        rows.append(
            {
                **scenario,
                "estimated_n_per_arm": n_per_arm,
                "arms": 4,
                "estimated_total_n": n_per_arm * 4,
                "status": "planning estimate only; not empirical evidence",
            }
        )
    frame = pd.DataFrame(rows)
    csv_path = out_dir / "power_analysis.csv"
    frame.to_csv(csv_path, index=False)
    doc_path = REPO_ROOT / "docs" / "prospective_trial" / "POWER_ANALYSIS.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "# Prospective ReliaGuard-NS Power Analysis\n\n"
        "This is a planning document, not a completed human-subject result. "
        "The default design is a four-arm randomized study comparing no gating, "
        "confidence-threshold gating, symbolic-rule gating and ReliaGuard-NS conformal gating.\n\n"
        + _markdown_table(frame)
        + "\n\nRecruitment must not begin until IRB/ethics approval, budget, consent, and recruitment credentials are confirmed.\n",
        encoding="utf-8",
    )
    return {"prospective_power_analysis": csv_path, "prospective_power_doc": doc_path}
