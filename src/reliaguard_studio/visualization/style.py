from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class NaturePalette:
    ink: str = "#17212B"
    muted: str = "#657386"
    grid: str = "#E3E8EF"
    panel: str = "#F7F9FC"
    paper: str = "#FFFFFF"
    navy: str = "#102A43"
    blue: str = "#2F6F9F"
    sky: str = "#8EC3DD"
    teal: str = "#1B998B"
    green: str = "#2D7D46"
    mint: str = "#A6D8C7"
    gold: str = "#C68E2C"
    sand: str = "#F2D7A0"
    rust: str = "#B75D32"
    red: str = "#B64045"
    rose: str = "#E9A1A1"
    violet: str = "#6C5B7B"
    lavender: str = "#D7C9E8"


PALETTE = NaturePalette()

DATASET_COLORS = {
    "HAIID": PALETTE.blue,
    "CHI 2023 DKE": PALETTE.gold,
    "ConvXAI": PALETTE.teal,
    "Pardos/Bhandari": PALETTE.rust,
    "FLoRA IPS": PALETTE.violet,
}

CONSTRUCT_COLORS = {
    "appropriate reliance": PALETTE.green,
    "overreliance": PALETTE.red,
    "underreliance": PALETTE.gold,
    "confidence/calibration": PALETTE.blue,
    "learning gain": PALETTE.rust,
    "process traces": PALETTE.violet,
    "unsupported": PALETTE.muted,
}

MODEL_COLORS = {
    "majority baseline": "#9AA6B2",
    "logistic regression": PALETTE.blue,
    "calibrated gradient boosting": PALETTE.teal,
    "symbolic-only": PALETTE.gold,
    "uncertainty-aware fusion": PALETTE.rust,
    "reliance-state neuro-symbolic": PALETTE.green,
}

POLICY_COLORS = {
    "Observed": PALETTE.muted,
    "Confidence threshold": PALETTE.gold,
    "Symbolic gating": PALETTE.blue,
    "Neuro-symbolic gating": PALETTE.green,
}


def apply_nature_style() -> None:
    """Apply a restrained, publication-oriented matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE.grid,
            "axes.linewidth": 0.8,
            "xtick.color": PALETTE.ink,
            "ytick.color": PALETTE.ink,
            "text.color": PALETTE.ink,
            "figure.facecolor": PALETTE.paper,
            "axes.facecolor": PALETTE.paper,
            "legend.frameon": False,
            "savefig.facecolor": PALETTE.paper,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": 1.6,
            "patch.linewidth": 1.0,
        }
    )


def clean_label(text: object) -> str:
    label = str(text)
    replacements = {
        "haiid": "HAIID",
        "chi2023_dke": "CHI 2023 DKE",
        "convxai_iui2025": "ConvXAI",
        "pardos_chatgpt_tutoring": "Pardos/Bhandari",
        "flora_ips": "FLoRA IPS",
        "xai": "XAI",
        "chatgpt": "ChatGPT",
        "chatxaiAuto": "LLM-agent conversational XAI",
        "chatxaiboost": "Boosted conversational XAI",
        "dashboard": "XAI dashboard",
        "control": "Control",
        "tutorial_present": "Tutorial",
        "xai_present": "XAI",
        "first_batch_overestimation": "Overestimation",
        "first_batch_underestimation": "Underestimation",
        "trust_first": "Initial trust",
        "post_explain_reliability": "Post-explanation reliability",
        "initial_confidence": "Initial confidence",
        "confidence_change": "Confidence shift",
        "overreliance": "Overreliance",
        "underreliance": "Underreliance",
        "appropriate_reliance": "Appropriate reliance",
        "reliance_state_neurosymbolic": "Reliance-state neuro-symbolic",
        "uncertainty_aware_fusion": "Uncertainty-aware fusion",
        "calibrated_gradient_boosting": "Calibrated gradient boosting",
        "logistic_regression": "Logistic regression",
        "symbolic_only": "Symbolic-only",
    }
    if label in replacements:
        return replacements[label]
    label = label.replace("C(advice_source_label)[T.human]", "Human-labelled advice")
    label = label.replace("C(condition_id)[T.chatxaiAuto]", "LLM-agent conversational XAI")
    label = label.replace("C(condition_id)[T.chatxaiboost]", "Boosted conversational XAI")
    label = label.replace("C(condition_id)[T.dashboard]", "XAI dashboard")
    label = label.replace("_", " ")
    return " ".join(part if part in {"AI", "XAI", "HAIID"} else part for part in label.split())

