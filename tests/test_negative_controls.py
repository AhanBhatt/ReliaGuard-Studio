from __future__ import annotations

import pandas as pd

from reliaguard_studio.evaluation import negative_controls


def test_negative_controls_write_label_and_advice_shuffle_outputs(tmp_path, monkeypatch):
    experiments = tmp_path / "experiments"
    prepared = tmp_path / "prepared"
    experiments.mkdir()
    dataset_dir = prepared / "toy"
    dataset_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "dataset": ["toy"] * 40,
            "target": ["overreliance"] * 40,
            "split": ["random"] * 40,
            "model": ["toy_model"] * 40,
            "y_true": [0, 1] * 20,
            "y_prob": [0.2, 0.8] * 20,
        }
    ).to_csv(experiments / "real_predictions.csv", index=False)
    pd.DataFrame(
        {
            "task_family": ["a"] * 20 + ["b"] * 20,
            "initial_correct": [1, 0] * 20,
            "final_correct": [0, 1] * 20,
            "advice_correct": [0, 1] * 20,
        }
    ).to_csv(dataset_dir / "interactions.csv", index=False)

    monkeypatch.setattr(negative_controls, "REAL_DATA_EXPERIMENTS_DIR", experiments)
    monkeypatch.setattr(negative_controls, "REAL_DATA_PREPARED_DIR", prepared)
    outputs = negative_controls.run_negative_controls(seed=1)
    result = pd.read_csv(outputs["negative_controls"])

    assert {"label_permutation", "advice_correctness_shuffle"}.issubset(set(result["control"]))
    assert "auroc_control" in result.columns
