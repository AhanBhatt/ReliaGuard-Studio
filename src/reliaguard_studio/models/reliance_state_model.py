from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RELIANCE_STATES = [
    "independent_correct",
    "independent_incorrect",
    "beneficial_ai_reliance",
    "harmful_overreliance",
    "harmful_underreliance",
    "correct_self_reliance",
    "confidence_inflated_reliance",
    "uncertain_disagreement",
]


@dataclass(frozen=True)
class RelianceStatePrediction:
    state_probabilities: dict[str, float]
    calibrated_risk: float
    uncertainty: float
    top_state: str


class RelianceStateModel:
    """Lightweight neuro-symbolic state layer used by the real-data pipeline.

    This class intentionally separates observable behavioural definitions from
    predictive risk. It can consume tabular probabilities, symbolic scores and
    uncertainty estimates produced elsewhere in the package and returns a
    normalized reliance-state distribution suitable for explanation and gating.
    """

    def infer_state_distribution(
        self,
        neural_probability: float,
        symbolic_probability: float,
        uncertainty: float,
        row: pd.Series | None = None,
    ) -> RelianceStatePrediction:
        row = row if row is not None else pd.Series(dtype=float)
        confidence = float(row.get("initial_confidence", 0.5) or 0.5)
        confidence_change = float(row.get("confidence_change", 0.0) or 0.0)
        advice_correct_proxy = float(row.get("advice_correct", row.get("post_explain_reliability", 0.5)) or 0.5)
        initial_correct_proxy = float(row.get("initial_correct", 0.5) or 0.5)

        harmful_risk = np.clip(0.45 * neural_probability + 0.35 * symbolic_probability + 0.20 * uncertainty, 0.0, 1.0)
        beneficial_risk = np.clip(0.45 * (1.0 - harmful_risk) + 0.35 * advice_correct_proxy + 0.20 * (1.0 - uncertainty), 0.0, 1.0)
        confidence_inflation = np.clip(max(confidence_change, 0.0) * confidence, 0.0, 1.0)

        raw = {
            "independent_correct": max(initial_correct_proxy * (1.0 - symbolic_probability), 0.01),
            "independent_incorrect": max((1.0 - initial_correct_proxy) * (1.0 - symbolic_probability), 0.01),
            "beneficial_ai_reliance": max(beneficial_risk * advice_correct_proxy, 0.01),
            "harmful_overreliance": max(harmful_risk * confidence * (1.0 - advice_correct_proxy), 0.01),
            "harmful_underreliance": max(harmful_risk * (1.0 - confidence) * advice_correct_proxy, 0.01),
            "correct_self_reliance": max(initial_correct_proxy * (1.0 - advice_correct_proxy) * confidence, 0.01),
            "confidence_inflated_reliance": max(confidence_inflation, 0.01),
            "uncertain_disagreement": max(uncertainty * (1.0 - abs(neural_probability - symbolic_probability)), 0.01),
        }
        total = sum(raw.values())
        probabilities = {state: float(value / total) for state, value in raw.items()}
        top_state = max(probabilities, key=probabilities.get)
        return RelianceStatePrediction(
            state_probabilities=probabilities,
            calibrated_risk=float(harmful_risk),
            uncertainty=float(uncertainty),
            top_state=top_state,
        )
