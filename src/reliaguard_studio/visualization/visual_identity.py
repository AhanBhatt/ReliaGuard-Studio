from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualIdentity:
    font: str = "Arial, Helvetica, sans-serif"
    ink: str = "#15212A"
    muted: str = "#5E6B73"
    faint: str = "#E8EEF2"
    panel_bg: str = "#F8FAFB"
    white: str = "#FFFFFF"
    accent: str = "#0B7285"
    warning: str = "#D9480F"
    good: str = "#2B8A3E"
    bad: str = "#C92A2A"
    uncertain: str = "#7048E8"
    learning: str = "#E67700"
    process: str = "#5C7CFA"
    line: float = 1.2
    radius: float = 12.0


VI = VisualIdentity()

DATASET_COLORS = {
    "HAIID": "#0B7285",
    "CHI 2023 DKE": "#5C7CFA",
    "ConvXAI": "#7048E8",
    "Pardos/Bhandari": "#E67700",
    "FLoRA IPS": "#2B8A3E",
}

CONSTRUCT_COLORS = {
    "Appropriate reliance": "#0B7285",
    "Overreliance": "#C92A2A",
    "Underreliance": "#E67700",
    "Confidence / calibration": "#5C7CFA",
    "Short-term learning gain": "#F59F00",
    "Process traces": "#2B8A3E",
    "Unsupported": "#868E96",
}

STATE_COLORS = {
    "Beneficial AI reliance": "#0B7285",
    "Beneficial AI Reliance": "#0B7285",
    "Correct self-reliance": "#2B8A3E",
    "Correct Self Reliance": "#2B8A3E",
    "Independent correct": "#66A80F",
    "Independent Correct": "#66A80F",
    "Independent incorrect": "#868E96",
    "Independent Incorrect": "#868E96",
    "Harmful overreliance": "#C92A2A",
    "Harmful AI Overreliance": "#C92A2A",
    "Harmful underreliance": "#E67700",
    "Harmful AI Underreliance": "#E67700",
    "Uncertain disagreement": "#7048E8",
    "Uncertain Disagreement": "#7048E8",
}

MODEL_COLORS = {
    "Logistic regression": "#7C8A96",
    "Calibrated gradient boosting": "#5C7CFA",
    "Symbolic-only": "#E67700",
    "Uncertainty-aware fusion": "#0B7285",
    "Reliance-state neuro-symbolic": "#7048E8",
}

POLICY_COLORS = {
    "Observed": "#7C8A96",
    "Confidence threshold": "#5C7CFA",
    "Symbolic gating": "#E67700",
    "Neuro-symbolic gating": "#7048E8",
}

DATASET_LABELS = {
    "haiid": "HAIID",
    "chi2023_dke": "CHI 2023 DKE",
    "convxai_iui2025": "ConvXAI",
    "pardos_chatgpt_tutoring": "Pardos/Bhandari",
    "flora_ips": "FLoRA IPS",
}

POLICY_LABELS = {
    "observed_no_gating": "Observed",
    "confidence_threshold_gating": "Confidence threshold",
    "symbolic_rule_gating": "Symbolic gating",
    "neurosymbolic_gating": "Neuro-symbolic gating",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "calibrated_logistic_regression": "Calibrated logistic regression",
    "calibrated_gradient_boosting": "Calibrated gradient boosting",
    "gradient_boosting": "Gradient boosting",
    "symbolic_only": "Symbolic-only",
    "weighted_fusion": "Weighted fusion",
    "uncertainty_aware_fusion": "Uncertainty-aware fusion",
    "learned_fusion": "Learned fusion",
    "reliance_state_neurosymbolic": "Reliance-state neuro-symbolic",
    "harmonized_logistic": "Harmonized logistic",
    "harmonized_gradient_boosting": "Harmonized gradient boosting",
}


def dataset_label(value: object) -> str:
    return DATASET_LABELS.get(str(value), human_label(value))


def model_label(value: object) -> str:
    return MODEL_LABELS.get(str(value), human_label(value))


def policy_label(value: object) -> str:
    return POLICY_LABELS.get(str(value), human_label(value))


def human_label(value: object) -> str:
    text = str(value).strip()
    replacements = {
        "ai": "AI",
        "xai": "XAI",
        "llm": "LLM",
        "dke": "DKE",
        "gee": "GEE",
        "auroc": "AUROC",
        "ece": "ECE",
        "r2": "R2",
        "chi2023": "CHI 2023",
        "convxai": "ConvXAI",
        "haiid": "HAIID",
        "flora": "FLoRA",
        "genai": "GenAI",
        "chatgpt": "ChatGPT",
    }
    parts = text.replace("_", " ").replace("-", " ").split()
    cleaned: list[str] = []
    for part in parts:
        key = part.lower()
        cleaned.append(replacements.get(key, part.capitalize()))
    return " ".join(cleaned)


def lighten(hex_color: str, amount: float = 0.82) -> str:
    color = hex_color.lstrip("#")
    rgb = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
    out = [round(v + (255 - v) * amount) for v in rgb]
    return "#" + "".join(f"{v:02x}" for v in out)
