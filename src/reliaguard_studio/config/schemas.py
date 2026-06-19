from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class IndicatorConfig(BaseModel):
    id: str
    name: str
    description: str
    direction: Literal["higher_is_riskier", "lower_is_riskier"]
    category: str
    source_note: str


class UsagePatternConfig(BaseModel):
    id: str
    description: str
    risk_level: Literal["protective", "cautionary", "high"]
    impact_score: float = Field(ge=0.0, le=1.0)
    detection_method: str


class TaskFamilyConfig(BaseModel):
    id: str
    name: str
    description: str
    difficulty_range: tuple[float, float]
    verification_demand: float = Field(ge=0.0, le=1.0)
    recall_demand: float = Field(ge=0.0, le=1.0)
    transfer_demand: float = Field(ge=0.0, le=1.0)


class AssistanceConditionConfig(BaseModel):
    id: str
    name: str
    description: str
    ai_helpfulness: float = Field(ge=0.0, le=1.0)
    citation_support: float = Field(ge=0.0, le=1.0)
    flaw_plausibility: float = Field(ge=0.0, le=1.0)
    verification_prompt: float = Field(ge=0.0, le=1.0)
    reflection_prompt: float = Field(ge=0.0, le=1.0)
    tutor_scaffolding: float = Field(ge=0.0, le=1.0)


class MetricConfig(BaseModel):
    id: str
    name: str
    description: str


class RuleAntecedent(BaseModel):
    feature: str
    relation: Literal["high", "low", "increasing", "decreasing", "persistent_high", "persistent_low"]
    threshold: float = Field(ge=0.0, le=1.0)


class RuleConfig(BaseModel):
    id: str
    name: str
    group: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=-1.0, le=1.0)
    temporal_window: PositiveInt
    targets: list[str]
    antecedents: list[RuleAntecedent]
    counterfactual_hints: list[str]


class ModelConfig(BaseModel):
    sequence_length: PositiveInt = 5
    bootstrap_samples: PositiveInt = 50
    hidden_dim: PositiveInt = 32
    dropout: float = Field(ge=0.0, le=0.9)


class SimulationConfig(BaseModel):
    seed: PositiveInt
    n_users: PositiveInt
    sessions_per_user: PositiveInt
    benchmark_repetitions: PositiveInt = 1


class ProjectMetadata(BaseModel):
    title: str
    short_name: str
    framing_note: str
    manuscript_title: str


class ProjectConfig(BaseModel):
    metadata: ProjectMetadata
    simulation: SimulationConfig
    model: ModelConfig
    targets: list[str]
    regression_targets: list[str]
    indicators: list[IndicatorConfig]
    usage_patterns: list[UsagePatternConfig]
    task_families: list[TaskFamilyConfig]
    assistance_conditions: list[AssistanceConditionConfig]
    metrics: list[MetricConfig]
    rules: list[RuleConfig]
