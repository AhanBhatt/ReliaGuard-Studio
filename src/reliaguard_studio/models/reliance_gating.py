from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


GatingAction = Literal[
    "accept_advice",
    "resist_advice",
    "request_verification",
    "show_uncertainty_cue",
    "compare_evidence",
    "delay_final_answer",
    "withhold_advice",
]


@dataclass(frozen=True)
class GatingDecision:
    action: GatingAction
    overreliance_risk: float
    underreliance_risk: float
    uncertainty: float
    explanation: str


class RelianceStateGatingModel:
    """Rule-grounded gating layer for conservative offline policy simulation.

    The scores are deployable proxies: they intentionally avoid using final
    correctness or the realized reliance label. Where advice correctness is
    not known at intervention time, reliability/confidence signals are used
    instead of oracle labels.
    """

    def score_row(self, row: pd.Series) -> GatingDecision:
        confidence = float(row.get("initial_confidence", row.get("pre_test_score", 0.5)) or 0.5)
        confidence_change = float(row.get("confidence_change", 0.0) or 0.0)
        reliability = float(row.get("post_explain_reliability", row.get("stated_accuracy_normalized", 0.5)) or 0.5)
        engagement = float(row.get("user_question_rate", 0.0) or 0.0) + 0.02 * float(row.get("chat_turn_count", 0.0) or 0.0)
        trust = float(row.get("trust_first", row.get("propensity_to_trust", 0.5)) or 0.5)

        over_score = np.clip(0.35 * confidence + 0.30 * reliability + 0.20 * max(confidence_change, 0) + 0.15 * trust - 0.25 * engagement, 0, 1)
        under_score = np.clip(0.40 * (1 - confidence) + 0.25 * (1 - trust) + 0.20 * (1 - reliability) + 0.15 * engagement, 0, 1)
        uncertainty = float(np.clip(1.0 - abs(over_score - under_score), 0.0, 1.0))

        if over_score >= 0.72 and over_score - under_score >= 0.10:
            return GatingDecision("request_verification", float(over_score), float(under_score), uncertainty, "High confidence/reliability signal with low engagement suggests verification before accepting advice.")
        if under_score >= 0.72 and under_score - over_score >= 0.10:
            return GatingDecision("compare_evidence", float(over_score), float(under_score), uncertainty, "Low confidence/trust profile suggests structured evidence comparison before rejecting advice.")
        if uncertainty >= 0.90 and max(over_score, under_score) >= 0.55:
            return GatingDecision("show_uncertainty_cue", float(over_score), float(under_score), uncertainty, "Reliance state is ambiguous; show uncertainty cue or delay finalization.")
        if over_score < 0.60 and under_score < 0.60:
            return GatingDecision("accept_advice", float(over_score), float(under_score), uncertainty, "Low estimated harmful-reliance risk.")
        return GatingDecision("delay_final_answer", float(over_score), float(under_score), uncertainty, "Moderate risk profile; slow finalization and ask for justification.")


def score_gating_frame(frame: pd.DataFrame) -> pd.DataFrame:
    model = RelianceStateGatingModel()
    rows = []
    for _, row in frame.iterrows():
        decision = model.score_row(row)
        rows.append(
            {
                "gating_action": decision.action,
                "overreliance_risk_proxy": decision.overreliance_risk,
                "underreliance_risk_proxy": decision.underreliance_risk,
                "uncertainty_proxy": decision.uncertainty,
                "gating_explanation": decision.explanation,
            }
        )
    return pd.DataFrame(rows, index=frame.index)
