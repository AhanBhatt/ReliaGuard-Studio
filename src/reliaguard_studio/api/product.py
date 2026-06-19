from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..paths import REAL_DATA_EXPERIMENTS_DIR


RelianceState = Literal[
    "beneficial_ai_reliance",
    "harmful_overreliance",
    "harmful_underreliance",
    "correct_self_reliance",
    "independent_correct",
    "independent_incorrect",
    "uncertain_disagreement",
]


class RelianceCase(BaseModel):
    initial_answer: str = Field(..., examples=["A"])
    initial_confidence: float = Field(0.5, ge=0.0, le=1.0)
    ai_advice: str = Field(..., examples=["B"])
    final_answer: str = Field(..., examples=["B"])
    ground_truth: str = Field(..., examples=["A"])
    task_context: str = "general decision support"
    advice_source: str = "AI"
    model_confidence: float = Field(0.75, ge=0.0, le=1.0)


class ReliancePrediction(BaseModel):
    state: RelianceState
    harmful_reliance_risk: float
    uncertainty: float
    state_probabilities: dict[str, float]
    active_rules: list[str]
    counterfactual: str
    recommended_action: str
    explanation: str
    safety_boundary: str


def _norm(value: str) -> str:
    return str(value).strip().casefold()


def _same(left: str, right: str) -> bool:
    return _norm(left) == _norm(right)


def _state_for_case(case: RelianceCase) -> RelianceState:
    initial_correct = _same(case.initial_answer, case.ground_truth)
    final_correct = _same(case.final_answer, case.ground_truth)
    advice_correct = _same(case.ai_advice, case.ground_truth)
    adopted_advice = _same(case.final_answer, case.ai_advice)
    disagreement = not _same(case.initial_answer, case.ai_advice)

    if disagreement and not initial_correct and advice_correct and adopted_advice:
        return "beneficial_ai_reliance"
    if disagreement and initial_correct and not advice_correct and adopted_advice:
        return "harmful_overreliance"
    if disagreement and not initial_correct and advice_correct and not adopted_advice:
        return "harmful_underreliance"
    if disagreement and initial_correct and not advice_correct and final_correct:
        return "correct_self_reliance"
    if final_correct:
        return "independent_correct"
    if not disagreement:
        return "independent_incorrect"
    return "uncertain_disagreement"


def predict_reliance_case(case: RelianceCase) -> ReliancePrediction:
    state = _state_for_case(case)
    initial_correct = _same(case.initial_answer, case.ground_truth)
    advice_correct = _same(case.ai_advice, case.ground_truth)
    adopted_advice = _same(case.final_answer, case.ai_advice)
    disagreement = not _same(case.initial_answer, case.ai_advice)

    confidence = case.initial_confidence
    advice_pressure = case.model_confidence if adopted_advice else 1.0 - case.model_confidence
    disagreement_boost = 0.18 if disagreement else -0.08
    over_component = 0.55 if state == "harmful_overreliance" else 0.0
    under_component = 0.50 if state == "harmful_underreliance" else 0.0
    risk = max(
        0.02,
        min(
            0.98,
            0.20
            + disagreement_boost
            + over_component
            + under_component
            + 0.16 * confidence
            + 0.10 * advice_pressure
            - (0.16 if state in {"beneficial_ai_reliance", "correct_self_reliance"} else 0.0),
        ),
    )
    uncertainty = max(0.03, min(0.95, 1.0 - abs(case.model_confidence - confidence)))

    active_rules: list[str] = []
    if disagreement:
        active_rules.append("human_advice_disagreement")
    if initial_correct and not advice_correct:
        active_rules.append("wrong_advice_overreliance_risk")
    if not initial_correct and advice_correct:
        active_rules.append("correct_advice_underuse_risk")
    if confidence >= 0.75:
        active_rules.append("high_initial_confidence")
    if adopted_advice:
        active_rules.append("advice_adoption")

    if state == "harmful_overreliance":
        action = "request_verification"
        counterfactual = "Risk would decrease if the user compared evidence before adopting the advice."
        explanation = "The user was initially correct, the advice was wrong, and the final answer moved toward the advice."
    elif state == "harmful_underreliance":
        action = "compare_evidence"
        counterfactual = "Risk would decrease if the user reviewed why the correct advice conflicts with the initial answer."
        explanation = "The user was initially wrong, the advice was correct, and the final answer did not adopt the correct advice."
    elif state == "correct_self_reliance":
        action = "accept_self_reliance"
        counterfactual = "Maintaining evidence comparison protects against wrong advice in similar cases."
        explanation = "The user resisted wrong advice and retained a correct answer."
    elif state == "beneficial_ai_reliance":
        action = "accept_advice_with_trace"
        counterfactual = "The case remains safer if the user can articulate why the advice is correct."
        explanation = "The user moved from an incorrect initial answer to the correct advice."
    else:
        action = "show_uncertainty_cue" if uncertainty > 0.65 else "monitor"
        counterfactual = "Risk would decrease with clearer evidence comparison or calibrated uncertainty cues."
        explanation = "The available tuple does not identify a sharper harmful or beneficial reliance transition."

    base = {
        "beneficial_ai_reliance": 0.06,
        "harmful_overreliance": 0.06,
        "harmful_underreliance": 0.06,
        "correct_self_reliance": 0.06,
        "independent_correct": 0.06,
        "independent_incorrect": 0.06,
        "uncertain_disagreement": 0.06,
    }
    base[state] = 0.58
    if disagreement:
        base["uncertain_disagreement"] += 0.08 * uncertainty
    total = sum(base.values())
    probabilities = {key: round(value / total, 3) for key, value in base.items()}

    return ReliancePrediction(
        state=state,
        harmful_reliance_risk=round(float(risk), 3),
        uncertainty=round(float(uncertainty), 3),
        state_probabilities=probabilities,
        active_rules=active_rules,
        counterfactual=counterfactual,
        recommended_action=action,
        explanation=explanation,
        safety_boundary="Evaluation support only: not a clinical, diagnostic, or causal deployment-effect claim.",
    )


def explain_case(case: RelianceCase) -> dict[str, Any]:
    prediction = predict_reliance_case(case)
    return {
        "plain_language_summary": prediction.explanation,
        "active_rules": prediction.active_rules,
        "counterfactual": prediction.counterfactual,
        "recommended_action": prediction.recommended_action,
        "ask_reliaguard_response": (
            f"ReliaGuard classified this as {prediction.state.replace('_', ' ')}. "
            f"The harmful-reliance risk is {prediction.harmful_reliance_risk:.2f}. "
            f"The recommendation is to {prediction.recommended_action.replace('_', ' ')}."
        ),
    }


def _read_csv(name: str) -> pd.DataFrame:
    path = REAL_DATA_EXPERIMENTS_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def dataset_summary() -> list[dict[str, Any]]:
    summary = _read_csv("reliaguard_dataset_summary.csv")
    if not summary.empty:
        return summary.to_dict(orient="records")
    return [
        {"dataset": "HAIID", "records": 35670, "participants": 1125},
        {"dataset": "CHI 2023 DKE", "records": 3984, "participants": 249},
        {"dataset": "ConvXAI", "records": 3060, "participants": 306},
        {"dataset": "Pardos/Bhandari", "records": 274, "participants": 274},
        {"dataset": "FLoRA IPS", "records": 275, "participants": 275},
    ]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _preview_conformal_rows(frame: pd.DataFrame, requested_alpha: float, served_alpha: float) -> pd.DataFrame:
    """Create a monotone UI preview when only one calibrated alpha is available.

    This is intentionally labelled as a preview, not as a fresh conformal
    guarantee. The exact calibrated artifact remains the scientific source of
    truth; the preview keeps the product dashboard interactive while preserving
    the evidence boundary.
    """
    preview = frame.copy()
    delta = float(served_alpha - requested_alpha)
    threshold_scale = max(0.35, min(1.85, (requested_alpha / max(served_alpha, 1e-6)) ** 0.55))

    if "threshold" in preview:
        preview["threshold"] = preview["threshold"].astype(float).map(lambda value: _clip01(value * threshold_scale))
    if "empirical_harmful_capture" in preview:
        preview["empirical_harmful_capture"] = preview["empirical_harmful_capture"].astype(float).map(lambda value: _clip01(value + 0.65 * delta))
    if "missed_harmful_fraction" in preview and "empirical_harmful_capture" in preview:
        preview["missed_harmful_fraction"] = preview["empirical_harmful_capture"].astype(float).map(lambda value: _clip01(1.0 - value))
    if "intervention_burden" in preview:
        preview["intervention_burden"] = preview["intervention_burden"].astype(float).map(lambda value: _clip01(value + 0.45 * delta))
    if "non_intervention_rate" in preview and "intervention_burden" in preview:
        preview["non_intervention_rate"] = preview["intervention_burden"].astype(float).map(lambda value: _clip01(1.0 - value))
    if "harmful_rate_among_non_intervened" in preview:
        preview["harmful_rate_among_non_intervened"] = preview["harmful_rate_among_non_intervened"].astype(float).map(lambda value: _clip01(value + 0.35 * (requested_alpha - served_alpha)))
    if "missed_harmful_bound" in preview:
        slack = preview.get("finite_sample_slack", pd.Series([0.0] * len(preview))).astype(float)
        preview["missed_harmful_bound"] = [_clip01(requested_alpha + s) for s in slack]

    preview["alpha"] = requested_alpha
    preview["served_artifact_alpha"] = served_alpha
    preview["alpha_source"] = "preview_estimate_from_nearest_artifact"
    preview["preview_note"] = (
        "Preview estimate for UI exploration. Re-run nsca run-conformal-risk-control "
        "to produce exact calibrated artifacts at this alpha."
    )
    return preview


def conformal_thresholds(alpha: float = 0.10) -> list[dict[str, Any]]:
    frame = _read_csv("reliaguard_conformal_results.csv")
    if frame.empty:
        return []
    requested_alpha = float(max(0.001, min(0.50, alpha)))
    served_alpha = requested_alpha
    alpha_source = "exact_artifact"
    if "alpha" in frame:
        frame["alpha"] = frame["alpha"].astype(float)
        available = sorted(frame["alpha"].dropna().unique().tolist())
        exact = frame.loc[(frame["alpha"] - requested_alpha).abs() < 1e-9]
        if exact.empty and available:
            served_alpha = min(available, key=lambda value: abs(value - requested_alpha))
            nearest = frame.loc[(frame["alpha"] - served_alpha).abs() < 1e-9]
            frame = _preview_conformal_rows(nearest, requested_alpha, served_alpha)
            alpha_source = "preview_estimate_from_nearest_artifact"
        else:
            frame = exact
            frame = frame.copy()
            frame["served_artifact_alpha"] = served_alpha
            frame["alpha_source"] = alpha_source
            frame["preview_note"] = "Exact calibrated artifact for this alpha."
    columns = [
        "dataset",
        "target",
        "model",
        "alpha",
        "served_artifact_alpha",
        "alpha_source",
        "threshold",
        "empirical_harmful_capture",
        "missed_harmful_fraction",
        "intervention_burden",
        "non_intervention_rate",
        "guarantee_scope",
        "preview_note",
    ]
    return frame[[c for c in columns if c in frame.columns]].to_dict(orient="records")


def policy_simulation() -> list[dict[str, Any]]:
    frame = _read_csv("policy_evaluation.csv")
    return frame.to_dict(orient="records") if not frame.empty else []


def ablation_results() -> list[dict[str, Any]]:
    frame = _read_csv("ablation_summary.csv")
    return frame.to_dict(orient="records") if not frame.empty else []


def model_card() -> dict[str, Any]:
    return {
        "model_name": "ReliaGuard-NS",
        "intended_use": "AI evaluation and selective-risk analysis for human-AI decision behavior.",
        "not_for": ["clinical diagnosis", "cognitive decline assessment", "causal deployment-effect claims"],
        "datasets": ["HAIID", "CHI 2023 DKE", "ConvXAI", "Pardos/Bhandari", "FLoRA IPS"],
        "outputs": ["reliance state", "harmful-reliance risk", "active rules", "counterfactual", "candidate action"],
        "limitations": [
            "No completed prospective intervention trial.",
            "Cross-dataset transfer is partial.",
            "Conformal gating can require high intervention burden.",
            "Learning/process datasets do not support delayed recall or transfer claims.",
        ],
    }


def evaluation_lab() -> dict[str, Any]:
    model_results = _read_csv("real_model_results.csv")
    calibration = _read_csv("calibration_summary.csv")
    transfer = _read_csv("cross_dataset_results.csv")
    return {
        "model_results": model_results.head(200).to_dict(orient="records") if not model_results.empty else [],
        "calibration": calibration.head(200).to_dict(orient="records") if not calibration.empty else [],
        "cross_dataset_transfer": transfer.head(200).to_dict(orient="records") if not transfer.empty else [],
    }
