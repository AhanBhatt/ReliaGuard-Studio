from __future__ import annotations

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR
from ..theory.reliance_formalism import calibration_utility_error_bound


def build_utility_bound_table(epsilons: list[float] | None = None, alpha: float = 2.0, beta: float = 1.0) -> pd.DataFrame:
    epsilons = epsilons or [0.01, 0.025, 0.05, 0.075, 0.10]
    return pd.DataFrame(
        [
            {
                "epsilon_calibration_error": epsilon,
                "alpha_overreliance_penalty": alpha,
                "beta_underreliance_penalty": beta,
                "absolute_utility_error_bound": calibration_utility_error_bound(epsilon, alpha=alpha, beta=beta),
                "note": "Bound applies to probability-driven utility error; intervention burden is deterministic conditional on the action.",
            }
            for epsilon in epsilons
        ]
    )


def write_utility_bound_table() -> dict[str, str]:
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration_utility_bounds.csv"
    build_utility_bound_table().to_csv(path, index=False)
    return {"calibration_utility_bounds": str(path)}

