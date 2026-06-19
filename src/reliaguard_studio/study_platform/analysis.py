from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def analyze_study_data() -> dict[str, Path]:
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "prospective_study"
    sim_path = out_dir / "simulated_study_data.csv"
    response_path = out_dir / "responses_export.csv"
    if sim_path.exists():
        data = pd.read_csv(sim_path)
    elif response_path.exists():
        data = pd.read_csv(response_path)
    else:
        raise FileNotFoundError("No prospective-study data found. Run `nsca simulate-study` or collect/export real study data first.")
    rows = []
    for condition, group in data.groupby("condition", dropna=False):
        row = {"condition": condition, "n_records": int(len(group))}
        for col in ["final_correct", "overreliance", "underreliance", "verification_action"]:
            if col in group.columns:
                row[f"mean_{col}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        row["evidence_status"] = "synthetic smoke test" if bool(group.get("simulated", pd.Series([False])).any()) else "collected prospective data"
        rows.append(row)
    summary = pd.DataFrame(rows)
    out = out_dir / "study_analysis_summary.csv"
    summary.to_csv(out, index=False)
    return {"study_analysis_summary": out}

