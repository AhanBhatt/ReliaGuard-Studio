from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REPO_ROOT
from .analysis import analyze_study_data
from .quality_checks import run_quality_checks
from .validation import validate_study_platform


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def generate_study_report() -> dict[str, Path]:
    """Generate a prospective validation readiness report.

    This report is intentionally a readiness artifact. It must not be cited as
    evidence of prospective human-subject efficacy unless real approved data
    are collected and marked as non-simulated.
    """

    validation = validate_study_platform()
    quality = run_quality_checks()
    try:
        analysis = analyze_study_data()
    except FileNotFoundError:
        analysis = {}
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "prospective_study"
    report_path = REPO_ROOT / "docs" / "prospective_trial" / "PROSPECTIVE_STUDY_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Prospective ReliaGuard-NS Validation Readiness Report",
        "",
        "**Status:** external-ready platform report. This is not a report of recruited human participants.",
        "",
        "## Recruitment gate",
        "",
        _markdown_table(pd.read_csv(validation["study_platform_validation"])),
        "",
        "## Quality checks",
        "",
        _markdown_table(pd.read_csv(quality["study_quality_checks"])),
    ]
    if analysis:
        parts.extend(
            [
                "",
                "## Simulated dry-run summary",
                "",
                "The following table is generated from simulated smoke-test data only.",
                "",
                _markdown_table(pd.read_csv(analysis["study_analysis_summary"])),
            ]
        )
    parts.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "Real recruitment remains blocked until IRB or ethics approval, consent approval, recruitment credentials, budget confirmation, privacy review, deployment URL and AI-advice-provider configuration are all present.",
        ]
    )
    report_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {"prospective_study_report": report_path, **validation, **quality, **analysis}
