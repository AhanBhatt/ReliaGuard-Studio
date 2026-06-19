from __future__ import annotations

import pandas as pd


def mixed_effects_status_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "Logistic mixed-effects model",
                "status": "not used as primary",
                "reason": "Python support for reliable binomial random-effects estimation is less stable than GEE for this artifact.",
                "replacement": "Participant-cluster GEE with robust covariance",
            },
            {
                "model": "Gaussian mixed-effects learning-gain model",
                "status": "not required",
                "reason": "Pardos/Bhandari has one participant-level row per learner, so participant random intercepts are not identifiable.",
                "replacement": "Gaussian GEE / robust linear model with topic and condition terms",
            },
            {
                "model": "Gaussian mixed-effects process model",
                "status": "not required",
                "reason": "FLoRA is prepared as one student-level process summary per learner for the main artifact.",
                "replacement": "Five-fold student-level regression and Gaussian GEE process-association model",
            },
        ]
    )
