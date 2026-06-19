from __future__ import annotations

import pandas as pd


HEURISTIC_WEIGHTS = {
    "cognitive_offloading_index": 0.18,
    "verification_robustness": -0.16,
    "copy_paste_dependence": 0.12,
    "source_checking_rate": -0.10,
    "calibration_error": 0.10,
    "retention_gap": 0.12,
    "transfer_gap": 0.12,
    "prompt_depth_score": -0.08,
    "rolling_offloading": 0.08,
    "offloading_trend": 0.08,
}


def legacy_heuristic_score(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.45, index=frame.index, dtype=float)
    for feature, weight in HEURISTIC_WEIGHTS.items():
        if feature in frame:
            score = score + frame[feature] * weight
    return score.clip(0.0, 1.0)
