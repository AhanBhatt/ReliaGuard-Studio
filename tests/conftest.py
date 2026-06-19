from __future__ import annotations

import pandas as pd
import pytest

from reliaguard_studio.config.loader import load_project_config
from reliaguard_studio.config.schemas import ProjectConfig
from reliaguard_studio.data.benchmark import build_air_bench_catalog
from reliaguard_studio.data.simulation import simulate_air_bench_dataset
from reliaguard_studio.models.fusion import build_symbolic_feature_frame
from reliaguard_studio.rules.engine import FuzzyTemporalRuleEngine


@pytest.fixture(scope="session")
def small_config() -> ProjectConfig:
    raw = load_project_config().model_dump()
    raw["simulation"]["n_users"] = 18
    raw["simulation"]["sessions_per_user"] = 6
    raw["model"]["bootstrap_samples"] = 5
    raw["model"]["hidden_dim"] = 8
    raw["model"]["dropout"] = 0.1
    return ProjectConfig.model_validate(raw)


@pytest.fixture(scope="session")
def small_dataset(small_config: ProjectConfig) -> dict[str, pd.DataFrame]:
    catalog = build_air_bench_catalog(small_config, tasks_per_family=4)
    return simulate_air_bench_dataset(small_config, catalog)


@pytest.fixture(scope="session")
def small_modeling_frame(small_config: ProjectConfig, small_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sessions = small_dataset["sessions"].copy().reset_index(drop=True)
    engine = FuzzyTemporalRuleEngine(small_config)
    evaluations = [engine.evaluate_row(row) for _, row in sessions.iterrows()]
    symbolic = build_symbolic_feature_frame(evaluations)
    for target in small_config.targets + small_config.regression_targets:
        symbolic[f"symbolic::{target}"] = [evaluation["target_scores"].get(target, 0.5) for evaluation in evaluations]
    group_columns = [column for column in symbolic.columns if column.startswith("rule_group::")]
    symbolic["top_rule_group"] = (
        symbolic[group_columns].abs().idxmax(axis=1).str.replace("rule_group::", "", regex=False)
        if group_columns
        else "none"
    )
    return pd.concat([sessions, symbolic], axis=1)
