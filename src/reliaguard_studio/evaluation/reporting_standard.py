from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..paths import PAPER_DIR, REAL_DATA_PREPARED_DIR


DISPLAY_NAMES = {
    "haiid": "HAIID",
    "chi2023_dke": "CHI 2023 DKE",
    "convxai_iui2025": "ConvXAI",
    "pardos_chatgpt_tutoring": "Pardos/Bhandari",
    "flora_ips": "FLoRA IPS",
}


@dataclass(frozen=True)
class RelianceStateDefinition:
    state: str
    applicability: str
    observable_definition: str
    interpretation_boundary: str


STATE_DEFINITIONS = [
    RelianceStateDefinition(
        "beneficial AI reliance",
        "decision records with initial answer, advice, final answer and ground truth",
        "initial answer is incorrect, advice is correct and final answer adopts or moves toward correct advice",
        "behavioural state; not a latent psychological diagnosis",
    ),
    RelianceStateDefinition(
        "harmful overreliance",
        "decision records with disagreement and advice correctness",
        "initial answer is correct, advice is wrong and final answer moves toward or adopts wrong advice",
        "behavioural reliance failure; not clinical or individual diagnosis",
    ),
    RelianceStateDefinition(
        "harmful underreliance",
        "decision records with disagreement and advice correctness",
        "initial answer is wrong, advice is correct and final answer fails to move toward or adopt correct advice",
        "behavioural reliance failure; can reflect many mechanisms",
    ),
    RelianceStateDefinition(
        "correct self-reliance",
        "decision records with initial answer, advice, final answer and ground truth",
        "initial answer is correct, advice is wrong and final answer remains correct",
        "appropriate resistance to wrong advice under observed labels",
    ),
    RelianceStateDefinition(
        "independent correctness",
        "decision records with initial/final correctness",
        "initial and final answers are correct without an applicable advice-correction transition",
        "not evidence that advice was irrelevant unless advice exposure is modelled",
    ),
    RelianceStateDefinition(
        "independent error",
        "decision records with initial/final correctness",
        "initial and final answers are wrong without an applicable beneficial-reliance transition",
        "does not identify the cognitive cause of error",
    ),
    RelianceStateDefinition(
        "uncertain disagreement",
        "records with disagreement but insufficient movement or confidence information",
        "human and advice disagree, but available public variables do not identify a sharper state",
        "explicitly marks construct limits",
    ),
]


CHECKLIST_ROWS = [
    ("Observable tuple", "Report h0, advice/support, h1, ground truth and context availability."),
    ("Applicability denominator", "Report the case denominator for each state, not only the total dataset size."),
    ("Initial and final accuracy", "Report both, with participant-level bootstrap intervals when participants repeat."),
    ("Advice correctness", "Report whether advice correctness is observed, derived or unavailable."),
    ("Overreliance", "Use only cases where initially correct users move toward wrong advice."),
    ("Underreliance", "Use only cases where initially wrong users fail to use correct advice."),
    ("Confidence and calibration", "Separate confidence movement from correctness and state labels."),
    ("Split design", "Report random, participant-aware and task/domain-aware splits where possible."),
    ("Calibration", "Report Brier score, ECE and reliability evidence for risk scores."),
    ("Selective-risk control", "Report capture and burden together; never report capture alone."),
    ("Boundary claims", "State unsupported constructs such as delayed recall, transfer or deployed causal effects."),
]


def _dataset_support_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_dir in sorted(REAL_DATA_PREPARED_DIR.glob("*")):
        interactions_path = dataset_dir / "interactions.csv"
        participants_path = dataset_dir / "participants.csv"
        if not interactions_path.exists():
            continue
        df = pd.read_csv(interactions_path, nrows=50)
        full_n = sum(1 for _ in interactions_path.open("r", encoding="utf-8")) - 1
        participant_n = None
        if participants_path.exists():
            participant_n = max(sum(1 for _ in participants_path.open("r", encoding="utf-8")) - 1, 0)
        cols = set(df.columns)
        supports_tuple = {"initial_label", "final_label", "correct_label"}.issubset(cols) and (
            "advice_label" in cols or "advice_correct" in cols
        )
        rows.append(
            {
                "dataset": DISPLAY_NAMES.get(dataset_dir.name, dataset_dir.name),
                "records": full_n,
                "participants_or_students": participant_n,
                "initial_judgment": "initial_label" in cols or "initial_correct" in cols,
                "advice_or_support": "advice_label" in cols or "condition_name" in cols or "help_condition" in cols,
                "final_judgment": "final_label" in cols or "final_correct" in cols,
                "ground_truth": "correct_label" in cols or "final_correct" in cols,
                "confidence_context": any("confidence" in c or "trust" in c for c in cols),
                "state_labels_supported": supports_tuple,
                "learning_gain_supported": {"pre_test_score", "post_test_score"}.issubset(cols)
                or "learning_gain" in cols,
                "process_traces_supported": dataset_dir.name == "flora_ips",
                "unsupported_constructs": "delayed recall; transfer; clinical/diagnostic inference",
            }
        )
    return rows


def write_reporting_standard() -> dict[str, Path]:
    submission_dir = PAPER_DIR / "nmi_submission"
    source_dir = submission_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = PAPER_DIR / "nmi_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    definitions = pd.DataFrame([row.__dict__ for row in STATE_DEFINITIONS])
    support = pd.DataFrame(_dataset_support_rows())
    checklist = pd.DataFrame(CHECKLIST_ROWS, columns=["item", "recommended_reporting"])

    reporting_path = source_dir / "reporting_standard.csv"
    definitions_path = source_dir / "reliance_state_definitions.csv"
    support_path = source_dir / "dataset_state_applicability.csv"
    checklist_path = submission_dir / "reliance_state_reporting_checklist.md"

    pd.concat(
        [
            definitions.assign(section="state definition"),
            checklist.rename(columns={"item": "state", "recommended_reporting": "observable_definition"}).assign(
                applicability="future reports",
                interpretation_boundary="reporting checklist",
                section="checklist",
            ),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(reporting_path, index=False)
    definitions.to_csv(definitions_path, index=False)
    support.to_csv(support_path, index=False)

    lines = [
        "# Reliance-State Reporting Checklist",
        "",
        "This checklist is intended for future human-AI collaboration studies that report reliance-state outcomes.",
        "",
    ]
    for item, recommendation in CHECKLIST_ROWS:
        lines.append(f"- **{item}:** {recommendation}")
    lines.extend(
        [
            "",
            "Boundary: these labels are behavioural descriptions from observable task tuples. They are not clinical, diagnostic, or latent psychological classifications.",
        ]
    )
    checklist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "reporting_standard": reporting_path,
        "state_definitions": definitions_path,
        "dataset_state_applicability": support_path,
        "checklist": checklist_path,
    }


if __name__ == "__main__":
    write_reporting_standard()
