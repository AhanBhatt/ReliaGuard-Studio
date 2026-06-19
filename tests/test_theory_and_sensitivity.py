from __future__ import annotations

import numpy as np
import pandas as pd

from reliaguard_studio.evaluation.policy_sensitivity import build_policy_burden_sensitivity
from reliaguard_studio.evaluation.selective_prediction import selective_prediction_curve
from reliaguard_studio.theory.reliance_formalism import (
    calibration_utility_error_bound,
    construct_portability_score,
    decompose_reliance,
)


def test_reliance_decomposition_and_utility_bound() -> None:
    frame = pd.DataFrame(
        {
            "initial_correct": [1, 0, 1, 0],
            "final_correct": [1, 1, 0, 0],
            "advice_correct": [0, 1, 0, 1],
            "overreliance": [0, 0, 1, 0],
            "underreliance": [0, 0, 0, 1],
            "correct_ai_reliance": [0, 1, 0, 0],
            "correct_self_reliance": [1, 0, 0, 0],
        }
    )
    decomp = decompose_reliance(frame)
    assert decomp.final_accuracy == 0.5
    assert decomp.harmful_overreliance == 0.25
    assert calibration_utility_error_bound(0.05, alpha=2.0, beta=1.0) == 0.2
    assert 0 <= construct_portability_score(0.8, 0.65) <= 1


def test_selective_prediction_and_policy_sensitivity() -> None:
    curve = selective_prediction_curve(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.55, 0.45]))
    assert {"coverage", "accuracy", "error_rate"}.issubset(curve.columns)
    policy = pd.DataFrame(
        {
            "dataset": ["tiny", "tiny"],
            "policy": ["observed_no_gating", "neurosymbolic_gating"],
            "expected_final_correct": [0.7, 0.75],
            "expected_overreliance": [0.1, 0.08],
            "expected_underreliance": [0.2, 0.18],
            "intervention_burden": [0.0, 0.4],
        }
    )
    sensitivity = build_policy_burden_sensitivity(policy, kappas=[0.0, 0.1])
    assert sensitivity["burden_penalty_kappa"].nunique() == 2
    assert "utility_gain_vs_observed" in sensitivity

