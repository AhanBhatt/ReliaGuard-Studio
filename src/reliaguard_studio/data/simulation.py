from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config.schemas import AssistanceConditionConfig, ProjectConfig, TaskFamilyConfig


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


@dataclass
class SyntheticUser:
    user_id: int
    baseline_ability: float
    verification_skill: float
    memory_strength: float
    transfer_skill: float
    reliance_tendency: float
    reflection_habit: float
    calibration_skill: float
    copy_paste_propensity: float
    age: int
    education_level: str
    ai_exposure_hours: float


def _sample_users(config: ProjectConfig, rng: np.random.Generator) -> list[SyntheticUser]:
    education_levels = ["high_school", "undergraduate", "graduate", "phd"]
    users: list[SyntheticUser] = []
    for user_id in range(1, config.simulation.n_users + 1):
        baseline_ability = _clip(rng.normal(0.58, 0.16))
        verification_skill = _clip(rng.normal(0.55, 0.18))
        memory_strength = _clip(rng.normal(0.57, 0.15))
        transfer_skill = _clip(rng.normal(0.56, 0.16))
        reliance_tendency = _clip(rng.beta(2.3, 2.0))
        reflection_habit = _clip(rng.normal(0.50, 0.18))
        calibration_skill = _clip(rng.normal(0.55, 0.17))
        copy_paste_propensity = _clip(rng.beta(2.1, 2.8))
        age = int(rng.integers(18, 55))
        education_level = education_levels[int(rng.integers(0, len(education_levels)))]
        ai_exposure_hours = round(float(rng.uniform(0.0, 6.0)), 2)
        users.append(
            SyntheticUser(
                user_id=user_id,
                baseline_ability=baseline_ability,
                verification_skill=verification_skill,
                memory_strength=memory_strength,
                transfer_skill=transfer_skill,
                reliance_tendency=reliance_tendency,
                reflection_habit=reflection_habit,
                calibration_skill=calibration_skill,
                copy_paste_propensity=copy_paste_propensity,
                age=age,
                education_level=education_level,
                ai_exposure_hours=ai_exposure_hours,
            )
        )
    return users


def _row_from_user(user: SyntheticUser) -> dict[str, Any]:
    return {
        "user_id": user.user_id,
        "baseline_ability": user.baseline_ability,
        "verification_skill": user.verification_skill,
        "memory_strength": user.memory_strength,
        "transfer_skill": user.transfer_skill,
        "reliance_tendency": user.reliance_tendency,
        "reflection_habit": user.reflection_habit,
        "calibration_skill": user.calibration_skill,
        "copy_paste_propensity": user.copy_paste_propensity,
        "age": user.age,
        "education_level": user.education_level,
        "ai_exposure_hours": user.ai_exposure_hours,
    }


def _simulate_session(
    user: SyntheticUser,
    task: pd.Series,
    condition: AssistanceConditionConfig,
    family: TaskFamilyConfig,
    session_idx: int,
    history: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    difficulty = float(task["difficulty"])
    prev = history[-3:]
    rolling_offloading_prev = float(np.mean([item["cognitive_offloading_index"] for item in prev])) if prev else user.reliance_tendency
    rolling_verification_prev = float(np.mean([item["verification_robustness"] for item in prev])) if prev else user.verification_skill
    rolling_recall_prev = float(np.mean([item["delayed_recall_score"] for item in prev])) if prev else user.memory_strength

    prompt_depth_score = _clip(
        0.18
        + 0.45 * user.reflection_habit
        + 0.18 * condition.reflection_prompt
        + 0.10 * family.transfer_demand
        - 0.12 * user.reliance_tendency
        + rng.normal(0.0, 0.06)
    )
    reflection_depth = _clip(0.55 * prompt_depth_score + 0.30 * condition.reflection_prompt + rng.normal(0.0, 0.05))
    copy_paste_dependence = _clip(
        0.10
        + 0.48 * user.copy_paste_propensity
        + 0.26 * condition.ai_helpfulness
        + 0.18 * user.reliance_tendency
        - 0.22 * condition.verification_prompt
        - 0.16 * prompt_depth_score
        + rng.normal(0.0, 0.07)
    )
    edit_distance_reliance = _clip(
        0.15 + 0.62 * copy_paste_dependence + 0.10 * condition.ai_helpfulness - 0.12 * prompt_depth_score + rng.normal(0.0, 0.05)
    )
    source_checking_rate = _clip(
        0.08
        + 0.46 * user.verification_skill
        + 0.30 * condition.citation_support
        + 0.26 * condition.verification_prompt
        - 0.16 * user.reliance_tendency
        + rng.normal(0.0, 0.08)
    )
    verification_robustness = _clip(
        0.12
        + 0.44 * user.verification_skill
        + 0.22 * source_checking_rate
        + 0.14 * reflection_depth
        + 0.06 * rolling_verification_prev
        - 0.26 * copy_paste_dependence
        + rng.normal(0.0, 0.07)
    )
    citation_support = _clip(condition.citation_support)
    cognitive_offloading_index = _clip(
        0.12
        + 0.42 * user.reliance_tendency
        + 0.16 * condition.ai_helpfulness
        + 0.18 * copy_paste_dependence
        + 0.08 * rolling_offloading_prev
        - 0.18 * verification_robustness
        - 0.10 * prompt_depth_score
        + rng.normal(0.0, 0.07)
    )
    flawed_answer_acceptance = _clip(
        0.05
        + 0.34 * condition.flaw_plausibility
        + 0.28 * user.reliance_tendency
        + 0.12 * copy_paste_dependence
        - 0.30 * verification_robustness
        - 0.12 * source_checking_rate
        + rng.normal(0.0, 0.06)
    )
    confidence = _clip(
        0.25
        + 0.28 * user.baseline_ability
        + 0.10 * condition.ai_helpfulness
        + 0.12 * copy_paste_dependence
        + 0.12 * user.calibration_skill
        - 0.14 * difficulty
        + rng.normal(0.0, 0.07)
    )

    immediate_prob = _clip(
        _sigmoid(
            -0.8
            + 2.4 * user.baseline_ability
            + 1.1 * condition.ai_helpfulness
            + 0.45 * prompt_depth_score
            + 0.20 * verification_robustness
            - 2.1 * difficulty
            - 1.3 * flawed_answer_acceptance
        )
    )
    immediate_success = int(rng.random() < immediate_prob)

    delayed_recall_score = _clip(
        0.12
        + 0.38 * user.memory_strength
        + 0.14 * condition.tutor_scaffolding
        + 0.16 * reflection_depth
        + 0.08 * rolling_recall_prev
        - 0.28 * cognitive_offloading_index
        - 0.18 * difficulty
        + rng.normal(0.0, 0.07)
    )
    delayed_recall_success = int(rng.random() < delayed_recall_score)
    transfer_score = _clip(
        0.10
        + 0.40 * user.transfer_skill
        + 0.16 * condition.tutor_scaffolding
        + 0.10 * prompt_depth_score
        - 0.24 * cognitive_offloading_index
        - 0.20 * difficulty
        + rng.normal(0.0, 0.08)
    )
    transfer_success = int(rng.random() < transfer_score)
    false_ai_detection = _clip(
        0.08
        + 0.32 * user.verification_skill
        + 0.18 * source_checking_rate
        + 0.18 * reflection_depth
        - 0.20 * user.reliance_tendency
        - 0.12 * confidence
        + rng.normal(0.0, 0.07)
    )
    verification_failure = int(
        (condition.flaw_plausibility > 0.6 and false_ai_detection < 0.45)
        or verification_robustness < 0.35
        or (citation_support > 0.6 and source_checking_rate < 0.20)
    )
    retention_gap = _clip(max(immediate_prob - delayed_recall_score, 0.0))
    transfer_gap = _clip(max(immediate_prob - transfer_score, 0.0))
    calibration_error = _clip(abs(confidence - immediate_prob))
    overreliance_score = _clip(
        0.28 * cognitive_offloading_index
        + 0.18 * copy_paste_dependence
        + 0.18 * flawed_answer_acceptance
        + 0.16 * (1.0 - verification_robustness)
        + 0.10 * calibration_error
        + 0.10 * transfer_gap
    )
    overreliance_risk = int(overreliance_score > 0.52)
    high_offloading_behavior = int(cognitive_offloading_index > 0.58)
    source_required = int(family.verification_demand > 0.6 or condition.citation_support > 0.5)
    intervention_benefit = _clip(
        0.15 * condition.tutor_scaffolding
        + 0.18 * condition.verification_prompt
        + 0.10 * condition.citation_support
        - 0.12 * condition.flaw_plausibility
        - 0.08 * user.reliance_tendency
        + 0.05 * prompt_depth_score
    )
    prompt_count = int(np.round(1 + 6 * prompt_depth_score + 1.5 * condition.tutor_scaffolding + rng.normal(0.0, 0.8)))
    prompt_count = max(prompt_count, 1)
    session_seconds = int(np.round(220 + 200 * difficulty + 120 * prompt_depth_score + 80 * source_checking_rate + rng.normal(0, 35)))

    component_driver_map = {
        "verification": 0.60 * (1.0 - verification_robustness) + 0.40 * flawed_answer_acceptance,
        "offloading": 0.70 * cognitive_offloading_index + 0.30 * copy_paste_dependence,
        "retention": retention_gap,
        "transfer": transfer_gap,
        "calibration": calibration_error,
    }
    latent_driver = max(component_driver_map, key=component_driver_map.get)

    rolling_offloading = _clip(float(np.mean([*([item["cognitive_offloading_index"] for item in prev]), cognitive_offloading_index])))
    rolling_verification = _clip(float(np.mean([*([item["verification_robustness"] for item in prev]), verification_robustness])))
    offloading_trend = _clip(0.5 + cognitive_offloading_index - rolling_offloading_prev)

    return {
        "user_id": user.user_id,
        "session_id": f"U{user.user_id:03d}_S{session_idx:02d}",
        "session_index": session_idx,
        "task_id": task["task_id"],
        "task_family": family.id,
        "condition_id": condition.id,
        "condition_name": condition.name,
        "difficulty": difficulty,
        "age": user.age,
        "education_level": user.education_level,
        "ai_exposure_hours": user.ai_exposure_hours,
        "baseline_ability": user.baseline_ability,
        "verification_skill": user.verification_skill,
        "memory_strength": user.memory_strength,
        "transfer_skill": user.transfer_skill,
        "reliance_tendency": user.reliance_tendency,
        "reflection_habit": user.reflection_habit,
        "calibration_skill": user.calibration_skill,
        "copy_paste_propensity": user.copy_paste_propensity,
        "prompt_count": prompt_count,
        "session_seconds": session_seconds,
        "prompt_depth_score": prompt_depth_score,
        "reflection_depth": reflection_depth,
        "copy_paste_dependence": copy_paste_dependence,
        "edit_distance_reliance": edit_distance_reliance,
        "source_checking_rate": source_checking_rate,
        "verification_robustness": verification_robustness,
        "citation_support": citation_support,
        "cognitive_offloading_index": cognitive_offloading_index,
        "flawed_answer_acceptance": flawed_answer_acceptance,
        "confidence": confidence,
        "immediate_success_probability": immediate_prob,
        "immediate_success": immediate_success,
        "delayed_recall_score": delayed_recall_score,
        "delayed_recall_success": delayed_recall_success,
        "transfer_score": transfer_score,
        "transfer_success": transfer_success,
        "false_ai_detection": false_ai_detection,
        "verification_failure": verification_failure,
        "retention_gap": retention_gap,
        "transfer_gap": transfer_gap,
        "calibration_error": calibration_error,
        "overreliance_score": overreliance_score,
        "overreliance_risk": overreliance_risk,
        "high_offloading_behavior": high_offloading_behavior,
        "source_required": source_required,
        "intervention_benefit": intervention_benefit,
        "rolling_offloading": rolling_offloading,
        "rolling_verification": rolling_verification,
        "offloading_trend": offloading_trend,
        "latent_driver": latent_driver,
    }


def simulate_air_bench_dataset(config: ProjectConfig, task_catalog: pd.DataFrame, seed: int | None = None) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed or config.simulation.seed)
    users = _sample_users(config, rng)
    task_lookup = {
        family.id: task_catalog.loc[task_catalog["family"] == family.id].reset_index(drop=True)
        for family in config.task_families
    }

    user_rows = [_row_from_user(user) for user in users]
    session_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for user in users:
        history: list[dict[str, Any]] = []
        for session_idx in range(1, config.simulation.sessions_per_user + 1):
            family = config.task_families[(session_idx - 1 + user.user_id) % len(config.task_families)]
            family_tasks = task_lookup[family.id]
            task = family_tasks.iloc[int(rng.integers(0, len(family_tasks)))]
            condition = config.assistance_conditions[int(rng.integers(0, len(config.assistance_conditions)))]
            session = _simulate_session(user, task, condition, family, session_idx, history, rng)
            session_rows.append(session)
            history.append(session)
            event_rows.extend(
                [
                    {
                        "session_id": session["session_id"],
                        "event_type": "prompt",
                        "event_value": session["prompt_count"],
                    },
                    {
                        "session_id": session["session_id"],
                        "event_type": "source_check",
                        "event_value": session["source_checking_rate"],
                    },
                    {
                        "session_id": session["session_id"],
                        "event_type": "copy_paste",
                        "event_value": session["copy_paste_dependence"],
                    },
                ]
            )
            user.reliance_tendency = _clip(
                user.reliance_tendency
                + 0.025 * (condition.ai_helpfulness - condition.verification_prompt - condition.tutor_scaffolding)
                + 0.040 * (session["cognitive_offloading_index"] - 0.50)
            )
            user.verification_skill = _clip(
                user.verification_skill
                + 0.030 * (condition.verification_prompt + session["source_checking_rate"] - 0.90)
                - 0.020 * session["flawed_answer_acceptance"]
            )
            user.memory_strength = _clip(
                user.memory_strength
                + 0.025 * (condition.tutor_scaffolding + session["reflection_depth"] - 0.90)
                - 0.025 * (session["cognitive_offloading_index"] - 0.50)
            )
            user.transfer_skill = _clip(
                user.transfer_skill
                + 0.025 * (condition.tutor_scaffolding + session["prompt_depth_score"] - 0.95)
                - 0.020 * (session["cognitive_offloading_index"] - 0.50)
            )

    sessions_df = pd.DataFrame(session_rows).sort_values(["user_id", "session_index"]).reset_index(drop=True)
    sessions_df["condition_group"] = sessions_df["condition_id"].replace(
        {
            "socratic_tutor": "intervention",
            "ai_with_citations": "intervention",
            "verification_required": "intervention",
            "reflection_before_reveal": "intervention",
            "standard_ai": "standard",
            "flawed_ai": "stress_test",
            "no_ai": "control",
        }
    )
    sessions_df["target_any_failure"] = (
        (1 - sessions_df["delayed_recall_success"])
        | (1 - sessions_df["transfer_success"])
        | sessions_df["verification_failure"]
        | sessions_df["overreliance_risk"]
    ).astype(int)
    return {
        "users": pd.DataFrame(user_rows),
        "sessions": sessions_df,
        "events": pd.DataFrame(event_rows),
        "tasks": task_catalog.copy(),
    }
