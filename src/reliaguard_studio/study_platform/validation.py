from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REPO_ROOT
from .gating_conditions import GATING_CONDITIONS
from .tasks import TASK_BANK


RECRUITMENT_ENV = [
    "IRB_APPROVED",
    "HUMAN_SUBJECTS_OK",
    "STUDY_RECRUITMENT_ENABLED",
    "PROLIFIC_API_TOKEN",
    "STUDY_BUDGET_CONFIRMED",
    "CONSENT_TEMPLATE_APPROVED",
    "DATA_PRIVACY_APPROVED",
    "RECRUITMENT_COUNTRY_ALLOWED",
    "STUDY_DEPLOYMENT_URL",
    "AI_ADVICE_PROVIDER_CONFIGURED",
]


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def validate_study_platform() -> dict[str, Path]:
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "prospective_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "check": "task_bank_non_empty",
            "status": "pass" if len(TASK_BANK) >= 6 else "fail",
            "detail": f"{len(TASK_BANK)} tasks available",
        },
        {
            "check": "four_randomized_conditions",
            "status": "pass" if len(GATING_CONDITIONS) == 4 else "fail",
            "detail": ", ".join(condition.condition_id for condition in GATING_CONDITIONS),
        },
    ]
    recruitment_ready = True
    for name in RECRUITMENT_ENV:
        value = os.environ.get(name)
        ready = bool(value) and value.lower() not in {"0", "false", "no"}
        recruitment_ready = recruitment_ready and ready
        rows.append({"check": f"env_{name}", "status": "present" if ready else "missing", "detail": "set" if ready else "not set"})
    rows.append(
        {
            "check": "recruitment_gate",
            "status": "closed" if not recruitment_ready else "open",
            "detail": "No participant recruitment is allowed unless all human-subjects and budget gates are present.",
        }
    )
    frame = pd.DataFrame(rows)
    csv_path = out_dir / "study_platform_validation.csv"
    frame.to_csv(csv_path, index=False)
    doc_path = REPO_ROOT / "docs" / "prospective_trial" / "STUDY_PLATFORM_VALIDATION.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "# Prospective Study Platform Validation\n\n"
        "This validation confirms that the platform can be dry-run locally and that recruitment is blocked unless required human-subjects gates are present.\n\n"
        + _markdown_table(frame)
        + "\n",
        encoding="utf-8",
    )
    return {"study_platform_validation": csv_path, "study_platform_validation_doc": doc_path}
