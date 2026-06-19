from __future__ import annotations

from pathlib import Path

from .policy_sensitivity import write_policy_sensitivity_outputs


def write_burden_sensitivity_outputs() -> dict[str, Path]:
    return {key: Path(value) for key, value in write_policy_sensitivity_outputs().items()}
