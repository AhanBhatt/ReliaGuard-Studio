from __future__ import annotations

import pandas as pd

from reliaguard_studio.evaluation.conformal_risk_control import split_conformal_threshold
from reliaguard_studio.models.reliaguard_ns import evaluate_reliaguard_predictions
from reliaguard_studio.study_platform.randomization import CONDITIONS, assign_condition
from reliaguard_studio.study_platform.simulation import simulate_prospective_trial


def test_split_conformal_threshold_controls_structure() -> None:
    threshold = split_conformal_threshold([0.1, 0.2, 0.8, 0.9, 0.95, 0.7], [0, 0, 1, 1, 1, 1], alpha=0.25, minimum_positive=2)
    assert threshold is not None
    assert 0.0 <= threshold.threshold <= 1.0
    assert threshold.n_positive_calibration == 4
    assert threshold.finite_sample_slack > 0


def test_reliaguard_predictions_smoke() -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "dataset": "haiid",
                "target": "overreliance",
                "split": "participant",
                "model": "reliance_state_neurosymbolic",
                "participant_id": f"p{i}",
                "task_instance_key": f"t{i}",
                "y_true": int(i % 3 == 0),
                "y_prob": 0.75 if i % 3 == 0 else 0.25,
            }
        )
    summary, actions = evaluate_reliaguard_predictions(pd.DataFrame(rows), alpha=0.10)
    assert not summary.empty
    assert not actions.empty
    assert "missed_harmful_bound" in summary.columns
    assert set(actions["reliaguard_action"]).issubset({"request_verification", "no_intervention"})


def test_study_randomization_and_simulation() -> None:
    assert assign_condition("participant-1") in CONDITIONS
    outputs = simulate_prospective_trial(n_participants=12, seed=7)
    assert outputs["simulated_study_data"].exists()
    data = pd.read_csv(outputs["simulated_study_data"])
    assert data["simulated"].all()
    assert {"overreliance", "underreliance", "final_correct"}.issubset(data.columns)

