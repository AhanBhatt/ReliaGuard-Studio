from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR
from .randomization import CONDITIONS, assign_condition
from .tasks import TASK_BANK


def simulate_prospective_trial(n_participants: int = 120, seed: int = 42) -> dict[str, Path]:
    """Create simulated smoke-test data for platform QA, not evidence."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_participants):
        participant_id = f"sim_{i:04d}"
        condition = assign_condition(participant_id)
        for task in TASK_BANK:
            initial_correct = rng.random() < 0.62
            flawed = rng.random() < 0.35
            advice_correct = not flawed
            if condition == "reliaguard_ns" and flawed:
                verification = rng.random() < 0.72
            elif condition == "symbolic_rule" and flawed:
                verification = rng.random() < 0.55
            elif condition == "confidence_threshold" and flawed:
                verification = rng.random() < 0.42
            else:
                verification = rng.random() < 0.25
            final_correct_prob = 0.58 + 0.20 * advice_correct + 0.10 * verification - 0.08 * flawed
            final_correct = rng.random() < np.clip(final_correct_prob, 0.05, 0.95)
            rows.append(
                {
                    "participant_id": participant_id,
                    "condition": condition,
                    "task_id": task.task_id,
                    "task_family": task.task_family,
                    "initial_correct": int(initial_correct),
                    "advice_correct": int(advice_correct),
                    "final_correct": int(final_correct),
                    "verification_action": int(verification),
                    "overreliance": int(initial_correct and not advice_correct and not final_correct),
                    "underreliance": int((not initial_correct) and advice_correct and not final_correct),
                    "simulated": True,
                    "evidence_status": "synthetic smoke test only; not human-subject evidence",
                }
            )
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "prospective_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "simulated_study_data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"simulated_study_data": path}

