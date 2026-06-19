from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatingCondition:
    condition_id: str
    display_name: str
    warning_rule: str
    evidence_boundary: str


GATING_CONDITIONS = [
    GatingCondition("no_gating", "No gating", "Advice shown without a model-triggered warning.", "Control arm."),
    GatingCondition("confidence_threshold", "Confidence-threshold gating", "Warn when confidence/advice disagreement crosses a fixed threshold.", "Heuristic comparator."),
    GatingCondition("symbolic_rule", "Symbolic-rule gating", "Warn when a hand-auditable reliance rule activates.", "Interpretable non-neural comparator."),
    GatingCondition("reliaguard_ns", "ReliaGuard-NS conformal gating", "Warn when conformal harmful-reliance score exceeds the calibrated threshold.", "Primary preregistered intervention arm."),
]


def condition_ids() -> list[str]:
    return [condition.condition_id for condition in GATING_CONDITIONS]
