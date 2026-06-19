from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR


def write_doubly_robust_status() -> dict[str, Path]:
    """Document when doubly robust OPE is and is not identifiable here.

    The public datasets do not log observed ReliaGuard-NS intervention
    propensities because the policy was not deployed. We therefore provide an
    eligibility/status artifact rather than a misleading doubly robust causal
    estimate.
    """
    path = REAL_DATA_EXPERIMENTS_DIR / "doubly_robust_policy_status.csv"
    pd.DataFrame(
        [
            {
                "status": "not estimated as causal policy value",
                "reason": "ReliaGuard-NS was not prospectively randomized or logged with propensities in the public datasets.",
                "in_repo_replacement": "condition contrasts, conformal selective-risk diagnostics, conservative observational policy-value bounds",
                "external_requirement": "prospective randomized gating trial with logged assignment propensities",
            }
        ]
    ).to_csv(path, index=False)
    return {"doubly_robust_policy_status": path}

