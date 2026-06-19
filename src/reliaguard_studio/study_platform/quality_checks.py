from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR
from .gating_conditions import GATING_CONDITIONS
from .tasks import TASK_BANK


def run_quality_checks(data_path: Path | None = None) -> dict[str, Path]:
    """Run deployment-readiness checks for simulated or exported study data."""

    out_dir = REAL_DATA_EXPERIMENTS_DIR / "prospective_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    if data_path is None:
        data_path = out_dir / "simulated_study_data.csv"
    rows: list[dict[str, str | int | float]] = []
    rows.append(
        {
            "check": "task_bank_size",
            "status": "pass" if len(TASK_BANK) >= 6 else "fail",
            "detail": f"{len(TASK_BANK)} preregistered tasks available",
        }
    )
    rows.append(
        {
            "check": "condition_count",
            "status": "pass" if len(GATING_CONDITIONS) == 4 else "fail",
            "detail": ", ".join(condition.condition_id for condition in GATING_CONDITIONS),
        }
    )
    if data_path.exists():
        frame = pd.read_csv(data_path)
        rows.append({"check": "data_file_present", "status": "pass", "detail": str(data_path.name)})
        required = {"participant_id", "condition", "task_id", "final_correct", "simulated"}
        missing = sorted(required.difference(frame.columns))
        rows.append(
            {
                "check": "required_columns",
                "status": "pass" if not missing else "fail",
                "detail": "none missing" if not missing else ", ".join(missing),
            }
        )
        counts = frame.groupby("condition").size().to_dict() if "condition" in frame.columns else {}
        rows.append(
            {
                "check": "all_conditions_observed",
                "status": "pass" if set(counts) == {c.condition_id for c in GATING_CONDITIONS} else "warn",
                "detail": str(counts),
            }
        )
        simulated = bool(frame.get("simulated", pd.Series([False])).astype(bool).any())
        rows.append(
            {
                "check": "evidence_label",
                "status": "pass" if simulated else "warn",
                "detail": "SIMULATED DRY-RUN - NOT HUMAN DATA" if simulated else "contains exported non-simulated rows",
            }
        )
    else:
        rows.append({"check": "data_file_present", "status": "warn", "detail": "no dry-run data file found"})
    out = out_dir / "study_quality_checks.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"study_quality_checks": out}
