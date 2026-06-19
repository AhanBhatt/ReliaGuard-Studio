from __future__ import annotations

import pandas as pd

from reliaguard_studio.evaluation import leakage_audit


def test_leakage_audit_writes_duplicate_and_split_outputs(tmp_path, monkeypatch):
    experiments = tmp_path / "experiments"
    prepared = tmp_path / "prepared"
    paper = tmp_path / "paper"
    dataset_dir = prepared / "toy"
    experiments.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    paper.mkdir(parents=True)

    pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2"],
            "task_instance_key": ["t1", "t1", "t2"],
            "initial_label": ["A", "A", "B"],
            "final_label": ["B", "B", "B"],
            "advice_label": ["B", "B", "A"],
        }
    ).to_csv(dataset_dir / "interactions.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "target": ["overreliance", "overreliance"],
            "model": ["m", "m"],
            "split": ["random", "participant"],
            "auroc": [0.90, 0.70],
        }
    ).to_csv(experiments / "real_model_results.csv", index=False)

    monkeypatch.setattr(leakage_audit, "REAL_DATA_EXPERIMENTS_DIR", experiments)
    monkeypatch.setattr(leakage_audit, "REAL_DATA_PREPARED_DIR", prepared)
    monkeypatch.setattr(leakage_audit, "PAPER_DIR", paper)
    outputs = leakage_audit.run_leakage_audit()
    result = pd.read_csv(outputs["leakage_audit"])

    assert {"duplicate_record", "split_robustness"}.issubset(set(result["audit"]))
    assert outputs["leakage_report"].exists()
