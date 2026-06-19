from __future__ import annotations

import pandas as pd
import pytest

from reliaguard_studio.evaluation.effect_sizes import odds_ratio_from_2x2
from reliaguard_studio.evaluation import policy_evaluation
from reliaguard_studio.models.reliance_gating import score_gating_frame


def test_reliance_gating_scores_actions_without_outcome_leakage() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["a", "b", "c"],
            "task_instance_key": ["t1", "t2", "t3"],
            "initial_confidence": [0.2, 0.9, 0.5],
            "stated_accuracy_normalized": [0.95, 0.45, 0.5],
            "post_explain_reliability": [0.95, 0.4, 0.5],
            "user_question_rate": [0.0, 0.8, 0.2],
        }
    )
    scored = score_gating_frame(frame)
    assert {"gating_action", "overreliance_risk_proxy", "underreliance_risk_proxy", "uncertainty_proxy"}.issubset(scored.columns)
    assert scored["gating_action"].isin(
        [
            "accept_advice",
            "resist_advice",
            "request_verification",
            "show_uncertainty_cue",
            "compare_evidence",
            "delay_final_answer",
            "withhold_advice",
        ]
    ).all()


def test_policy_evaluation_smoke_with_tiny_fixture(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_evaluation, "REAL_DATA_EXPERIMENTS_DIR", tmp_path)
    frame = pd.DataFrame(
        {
            "participant_id": ["a", "b", "c", "d"],
            "task_instance_key": ["t1", "t2", "t3", "t4"],
            "disagreement_case": [1, 1, 1, 1],
            "final_correct": [1, 0, 1, 0],
            "overreliance": [0, 1, 0, 0],
            "underreliance": [0, 0, 1, 0],
            "initial_confidence": [0.2, 0.9, 0.7, 0.4],
        }
    )
    outputs = policy_evaluation.run_policy_evaluation({"tiny": frame})
    assert outputs["policy_evaluation"].exists()
    policies = pd.read_csv(outputs["policy_evaluation"])
    assert {"observed_no_gating", "neurosymbolic_gating"}.issubset(set(policies["policy"]))


def test_odds_ratio_effect_size_is_finite_with_smoothing() -> None:
    result = odds_ratio_from_2x2(a=10, b=0, c=2, d=8)
    assert result["odds_ratio"] > 1
    assert result["ci_low"] > 0
