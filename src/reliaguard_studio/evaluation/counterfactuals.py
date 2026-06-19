from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CounterfactualRecommendation:
    state: str
    action: str
    rationale: str
    minimally_changed_signal: str


def recommend_counterfactual(state: str, overreliance_risk: float, underreliance_risk: float, uncertainty: float) -> CounterfactualRecommendation:
    """Map reliance-state risks to an interpretable minimal intervention.

    The output is a decision-support explanation for offline analysis. It does
    not assert that the intervention would causally change the participant's
    behaviour without a prospective randomized test.
    """

    if overreliance_risk >= 0.70 and overreliance_risk >= underreliance_risk:
        return CounterfactualRecommendation(
            state=state,
            action="request_verification",
            rationale="High overreliance risk is most directly addressed by requiring evidence comparison before accepting advice.",
            minimally_changed_signal="increase source/evidence checking before final answer",
        )
    if underreliance_risk >= 0.70:
        return CounterfactualRecommendation(
            state=state,
            action="compare_evidence",
            rationale="High underreliance risk suggests the participant may reject correct advice without structured comparison.",
            minimally_changed_signal="make the advice rationale explicit and compare it with the initial answer",
        )
    if uncertainty >= 0.85:
        return CounterfactualRecommendation(
            state=state,
            action="show_uncertainty_cue",
            rationale="The model is uncertain about the reliance state, so an uncertainty cue is safer than a directive intervention.",
            minimally_changed_signal="display calibrated uncertainty and delay finalization",
        )
    return CounterfactualRecommendation(
        state=state,
        action="accept_advice",
        rationale="No high-risk reliance state is detected under the conservative proxy model.",
        minimally_changed_signal="no additional intervention",
    )
