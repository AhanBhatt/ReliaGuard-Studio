from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models.reliance_gating import score_gating_frame
from ..paths import REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR
from .metrics import bootstrap_confidence_interval
from .policy_sensitivity import write_policy_sensitivity_outputs
from .utility_bounds import write_utility_bound_table


def _load_prepared_interactions() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for dataset_dir in REAL_DATA_PREPARED_DIR.iterdir():
        path = dataset_dir / "interactions.csv"
        if path.exists():
            frames[dataset_dir.name] = pd.read_csv(path, low_memory=False)
    return frames


def _utility(final_correct: pd.Series, overreliance: pd.Series, underreliance: pd.Series, burden: float) -> float:
    return float(final_correct.mean() - 2.0 * overreliance.fillna(0).mean() - 1.0 * underreliance.fillna(0).mean() - 0.05 * burden)


def _simulate_policy(frame: pd.DataFrame, policy_name: str, actions: pd.Series) -> dict[str, Any]:
    over = frame.get("overreliance", pd.Series(0, index=frame.index)).fillna(0).astype(float).copy()
    under = frame.get("underreliance", pd.Series(0, index=frame.index)).fillna(0).astype(float).copy()
    final = frame.get("final_correct", pd.Series(np.nan, index=frame.index)).astype(float).copy()

    request_verification = actions.isin(["request_verification", "withhold_advice"])
    compare_evidence = actions.isin(["compare_evidence", "delay_final_answer"])
    uncertainty = actions.isin(["show_uncertainty_cue"])
    burden = float((request_verification | compare_evidence | uncertainty).mean())

    # Conservative observational simulation, not a causal estimate:
    # verification reduces realized overreliance by 20%, evidence comparison reduces
    # underreliance by 18%, uncertainty cues apply smaller symmetric reductions.
    adjusted_over = over.copy()
    adjusted_under = under.copy()
    adjusted_final = final.copy()
    adjusted_over.loc[request_verification] *= 0.80
    adjusted_under.loc[compare_evidence] *= 0.82
    adjusted_over.loc[uncertainty] *= 0.92
    adjusted_under.loc[uncertainty] *= 0.92
    harmful_reduction = (over - adjusted_over) + (under - adjusted_under)
    adjusted_final = np.clip(adjusted_final + 0.25 * harmful_reduction, 0.0, 1.0)

    utility_value = _utility(adjusted_final, adjusted_over, adjusted_under, burden)
    values = adjusted_final - 2.0 * adjusted_over - adjusted_under - 0.05 * (request_verification | compare_evidence | uncertainty).astype(float)
    low, high = bootstrap_confidence_interval(lambda a, b: float(np.mean(a)), values.to_numpy(dtype=float), values.to_numpy(dtype=float), n_boot=120, seed=42)
    return {
        "policy": policy_name,
        "n": int(len(frame)),
        "expected_final_correct": float(adjusted_final.mean()),
        "expected_overreliance": float(adjusted_over.mean()),
        "expected_underreliance": float(adjusted_under.mean()),
        "intervention_burden": burden,
        "expected_utility": utility_value,
        "utility_ci_low": low,
        "utility_ci_high": high,
        "note": "Conservative observational utility simulation; not a causal off-policy estimate.",
    }


def run_policy_evaluation(interactions_map: dict[str, pd.DataFrame] | None = None) -> dict[str, Path]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    interactions_map = interactions_map or _load_prepared_interactions()
    rows: list[dict[str, Any]] = []
    action_rows: list[pd.DataFrame] = []
    for dataset, frame in interactions_map.items():
        if dataset == "pardos_chatgpt_tutoring":
            continue
        needed = {"final_correct", "overreliance", "underreliance"}
        if not needed.issubset(frame.columns):
            continue
        analysis = frame.loc[frame.get("disagreement_case", 1) == 1].copy()
        if analysis.empty:
            analysis = frame.copy()
        scored = pd.concat([analysis.reset_index(drop=True), score_gating_frame(analysis).reset_index(drop=True)], axis=1)
        scored["dataset"] = dataset
        action_rows.append(scored[["dataset", "participant_id", "task_instance_key", "gating_action", "overreliance_risk_proxy", "underreliance_risk_proxy", "uncertainty_proxy"]])

        rows.append({"dataset": dataset, **_simulate_policy(scored, "observed_no_gating", pd.Series("accept_advice", index=scored.index))})
        confidence = scored["initial_confidence"].fillna(0.5).astype(float) if "initial_confidence" in scored.columns else pd.Series(0.5, index=scored.index)
        threshold_actions = pd.Series(np.where(confidence > 0.75, "request_verification", "accept_advice"), index=scored.index)
        rows.append({"dataset": dataset, **_simulate_policy(scored, "confidence_threshold_gating", threshold_actions)})
        symbolic_actions = pd.Series(np.where(scored["overreliance_risk_proxy"] > 0.66, "request_verification", np.where(scored["underreliance_risk_proxy"] > 0.66, "compare_evidence", "accept_advice")), index=scored.index)
        rows.append({"dataset": dataset, **_simulate_policy(scored, "symbolic_rule_gating", symbolic_actions)})
        rows.append({"dataset": dataset, **_simulate_policy(scored, "neurosymbolic_gating", scored["gating_action"])})

    policy_df = pd.DataFrame(rows)
    actions_df = pd.concat(action_rows, ignore_index=True) if action_rows else pd.DataFrame()
    policy_path = REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv"
    actions_path = REAL_DATA_EXPERIMENTS_DIR / "policy_actions.csv"
    policy_df.to_csv(policy_path, index=False)
    actions_df.to_csv(actions_path, index=False)
    sensitivity_outputs = write_policy_sensitivity_outputs(policy_df)
    bound_outputs = write_utility_bound_table()
    return {"policy_evaluation": policy_path, "policy_actions": actions_path, **{key: Path(value) for key, value in sensitivity_outputs.items()}, **{key: Path(value) for key, value in bound_outputs.items()}}
