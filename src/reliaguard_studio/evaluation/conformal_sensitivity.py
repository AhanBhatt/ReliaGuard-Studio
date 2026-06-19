from __future__ import annotations

from pathlib import Path

from .alpha_sensitivity import write_alpha_sensitivity_outputs
from .burden_sensitivity import write_burden_sensitivity_outputs
from .calibration_sensitivity import write_calibration_sensitivity_outputs
from .harm_weight_sensitivity import write_harm_weight_sensitivity_outputs
from .label_definition_sensitivity import write_label_definition_sensitivity_outputs
from .policy_value_bounds import run_policy_bound_suite
from .threshold_sensitivity import write_threshold_sensitivity_outputs


def run_sensitivity_analyses() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs.update(write_alpha_sensitivity_outputs())
    outputs.update(write_burden_sensitivity_outputs())
    outputs.update(write_label_definition_sensitivity_outputs())
    outputs.update(write_calibration_sensitivity_outputs())
    outputs.update(write_harm_weight_sensitivity_outputs())
    outputs.update(write_threshold_sensitivity_outputs())
    outputs.update(run_policy_bound_suite())
    return outputs
