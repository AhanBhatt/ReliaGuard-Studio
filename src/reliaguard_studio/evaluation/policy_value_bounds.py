from __future__ import annotations

from pathlib import Path

from .doubly_robust_policy import write_doubly_robust_status
from .net_benefit import write_net_benefit_table
from .off_policy_evaluation import run_off_policy_evaluation
from .sensitivity_to_confounding import write_confounding_sensitivity_grid


def run_policy_bound_suite() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs.update(run_off_policy_evaluation())
    outputs.update(write_doubly_robust_status())
    outputs.update(write_confounding_sensitivity_grid())
    outputs.update(write_net_benefit_table())
    return outputs

