from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RelianceDecomposition:
    initial_accuracy: float
    final_accuracy: float
    independent_correct: float
    independent_incorrect: float
    beneficial_ai_reliance: float
    correct_self_reliance: float
    harmful_overreliance: float
    harmful_underreliance: float

    @property
    def final_accuracy_identity(self) -> float:
        return self.independent_correct + self.beneficial_ai_reliance + self.correct_self_reliance

    @property
    def residual_error_identity(self) -> float:
        return self.independent_incorrect + self.harmful_overreliance + self.harmful_underreliance

    @property
    def gain(self) -> float:
        return self.final_accuracy - self.initial_accuracy


def decompose_reliance(frame: pd.DataFrame) -> RelianceDecomposition:
    """Compute the accuracy/reliance identity from observable interaction labels.

    The identity is descriptive. It does not assume that advice caused the final
    decision; it partitions observed records into mutually interpretable states
    whenever the canonical schema provides initial/final correctness and advice
    correctness.
    """
    if frame.empty:
        return RelianceDecomposition(*(float("nan") for _ in range(8)))
    n = float(len(frame))
    initial = frame.get("initial_correct", pd.Series(np.nan, index=frame.index)).astype(float)
    final = frame.get("final_correct", pd.Series(np.nan, index=frame.index)).astype(float)
    advice = frame.get("advice_correct", pd.Series(np.nan, index=frame.index)).astype(float)
    over = frame.get("overreliance", pd.Series(0, index=frame.index)).fillna(0).astype(float)
    under = frame.get("underreliance", pd.Series(0, index=frame.index)).fillna(0).astype(float)
    correct_ai = frame.get("correct_ai_reliance", pd.Series(0, index=frame.index)).fillna(0).astype(float)
    correct_self = frame.get("correct_self_reliance", pd.Series(0, index=frame.index)).fillna(0).astype(float)

    independent_correct = ((initial == 1) & (final == 1) & (advice == 1)).mean()
    independent_incorrect = ((final == 0) & (over == 0) & (under == 0)).mean()
    return RelianceDecomposition(
        initial_accuracy=float(initial.mean()),
        final_accuracy=float(final.mean()),
        independent_correct=float(independent_correct),
        independent_incorrect=float(independent_incorrect),
        beneficial_ai_reliance=float(correct_ai.mean()),
        correct_self_reliance=float(correct_self.mean()),
        harmful_overreliance=float(over.mean()),
        harmful_underreliance=float(under.mean()),
    )


def final_accuracy_decomposition_statement(decomp: RelianceDecomposition, tolerance: float = 0.05) -> str:
    gap = abs(decomp.final_accuracy - decomp.final_accuracy_identity)
    if np.isnan(gap):
        return "Decomposition unavailable because canonical labels are missing."
    if gap <= tolerance:
        return (
            "Final accuracy is approximately decomposed into independent correctness, "
            "beneficial AI reliance and correct self-reliance; residual error contains "
            "independent errors plus harmful overreliance and underreliance."
        )
    return (
        "Observed labels are not mutually exhaustive under the available schema; "
        "the decomposition should be interpreted as a lower-bound state partition."
    )


def calibration_utility_error_bound(epsilon: float, alpha: float = 2.0, beta: float = 1.0) -> float:
    """Bound utility error induced by epsilon-calibrated state probabilities.

    If probabilities for correctness, overreliance and underreliance are each
    calibrated within epsilon and utility is
    U=P(correct)-alpha P(over)-beta P(under)-kappa I(a), then the absolute
    probability-driven utility error is bounded by (1+alpha+beta) epsilon.
    The intervention-burden term is deterministic for a chosen action.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    return float((1.0 + alpha + beta) * epsilon)


def conformal_missed_harmful_bound(alpha: float, n_positive_calibration: int) -> float:
    """Finite-sample missed-harmful bound used by ReliaGuard-NS.

    The statement assumes exchangeability between calibration and deployment
    harmful examples. It is a selective-risk bound for the score threshold, not
    a causal effect bound for the intervention.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if n_positive_calibration < 1:
        raise ValueError("n_positive_calibration must be positive")
    return float(min(1.0, alpha + 1.0 / (n_positive_calibration + 1)))


def construct_portability_score(source_auc: float, target_auc: float, chance: float = 0.5) -> float:
    """Score how much above-chance discrimination transfers to a target dataset."""
    source_margin = max(source_auc - chance, 1e-12)
    target_margin = max(target_auc - chance, 0.0)
    return float(np.clip(target_margin / source_margin, 0.0, 1.0))
