from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalThreshold:
    threshold: float
    alpha: float
    n_calibration: int
    n_positive_calibration: int
    finite_sample_slack: float


def deterministic_split_key(*parts: object) -> float:
    """Return a deterministic pseudo-random value in [0, 1)."""
    text = "::".join(str(part) for part in parts)
    digest = sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_conformal_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    alpha: float = 0.10,
    minimum_positive: int = 5,
) -> ConformalThreshold | None:
    """One-sided split-conformal threshold for harmful-reliance detection.

    The threshold is the lower conformal quantile of scores observed among
    harmful examples in the calibration fold. Flagging cases with scores above
    this threshold targets high recall for harmful reliance. Under exchangeable
    calibration and deployment harmful examples, the missed-harmful fraction is
    bounded by approximately alpha plus the finite-sample slack reported here.
    """
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    positive_scores = np.sort(s[y == 1])
    n_pos = int(len(positive_scores))
    if n_pos < minimum_positive:
        return None
    rank = int(np.ceil((n_pos + 1) * alpha)) - 1
    rank = int(np.clip(rank, 0, n_pos - 1))
    threshold = float(positive_scores[rank])
    return ConformalThreshold(
        threshold=threshold,
        alpha=float(alpha),
        n_calibration=int(len(s)),
        n_positive_calibration=n_pos,
        finite_sample_slack=float(1.0 / (n_pos + 1)),
    )


def summarize_selective_risk(
    scores: pd.Series,
    labels: pd.Series,
    threshold: float,
    intervention_penalty: float = 0.05,
) -> dict[str, float]:
    y = labels.astype(int)
    score = scores.astype(float)
    intervene = score >= threshold
    allowed = ~intervene
    positives = y == 1
    missed_harmful = allowed & positives
    non_intervention_rate = float(allowed.mean()) if len(allowed) else 0.0
    burden = float(intervene.mean()) if len(intervene) else 0.0
    harmful_rate_allowed = float(y.loc[allowed].mean()) if allowed.any() else 0.0
    missed_harmful_fraction = float(missed_harmful.sum() / max(1, positives.sum()))
    utility = float((1.0 - y.loc[allowed]).mean() * non_intervention_rate - intervention_penalty * burden) if len(y) else 0.0
    return {
        "n_test": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "threshold": float(threshold),
        "non_intervention_rate": non_intervention_rate,
        "intervention_burden": burden,
        "harmful_rate_among_non_intervened": harmful_rate_allowed,
        "missed_harmful_fraction": missed_harmful_fraction,
        "empirical_harmful_capture": float(1.0 - missed_harmful_fraction),
        "expected_utility_proxy": utility,
    }


def assign_conformal_actions(
    frame: pd.DataFrame,
    threshold: float,
    target: str,
) -> pd.DataFrame:
    output = frame.copy()
    score = output["risk_score"].astype(float)
    intervene = score >= threshold
    if target == "overreliance":
        intervention = "request_verification"
        rule = "wrong-advice overreliance risk"
        counterfactual = "Ask the participant to verify evidence before accepting advice."
    elif target == "underreliance":
        intervention = "compare_evidence"
        rule = "correct-advice underreliance risk"
        counterfactual = "Ask the participant to compare their answer with the evidence supporting the advice."
    else:
        intervention = "show_uncertainty_cue"
        rule = "general harmful-reliance risk"
        counterfactual = "Display an uncertainty cue and slow finalization."
    output["conformal_threshold"] = float(threshold)
    output["reliaguard_action"] = np.where(intervene, intervention, "no_intervention")
    output["selective_action_set"] = np.where(intervene, intervention + "|delay_final_answer", "accept_or_continue")
    output["rule_trace"] = np.where(intervene, rule, "risk below conformal threshold")
    output["counterfactual_explanation"] = np.where(intervene, counterfactual, "No counterfactual shown because estimated harmful-reliance risk is below threshold.")
    return output

