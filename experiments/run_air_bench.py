from __future__ import annotations

from reliaguard_studio.config.loader import load_project_config
from reliaguard_studio.evaluation.runner import run_full_experiment


if __name__ == "__main__":
    run_full_experiment(load_project_config())
