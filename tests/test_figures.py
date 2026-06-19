from __future__ import annotations

import os

import pytest

from reliaguard_studio.paths import PAPER_FIGURES_DIR, REAL_DATA_EXPERIMENTS_DIR
from reliaguard_studio.visualization.real_data_figures import generate_real_data_figures
from reliaguard_studio.visualization.tables import generate_paper_tables


def test_figure_and_table_generation_smoke() -> None:
    if os.environ.get("NSCA_RUN_PRIVATE_PAPER_TESTS") != "1":
        pytest.skip("Private paper figure generation is skipped in the public/product test suite.")
    if not (REAL_DATA_EXPERIMENTS_DIR / "real_experiment_summary.json").exists():
        pytest.skip("real-data experiment outputs are required for real figure smoke test")

    table_outputs = generate_paper_tables()
    figure_outputs = generate_real_data_figures()

    assert table_outputs["main_model_policy_summary"].exists()
    assert any(path.name == "figure_01_evidence_to_action_map.pdf" for path in figure_outputs)
    assert (PAPER_FIGURES_DIR / "figure_06_rules_to_policy_frontier.pdf").exists()
