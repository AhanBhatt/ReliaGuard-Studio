from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config.schemas import ProjectConfig
from ..data.benchmark import build_air_bench_catalog
from ..data.simulation import simulate_air_bench_dataset
from ..models.baselines import (
    BaselineBundle,
    bootstrap_classification_uncertainty,
    get_feature_columns,
    train_classification_baselines,
    train_regression_baselines,
)
from ..models.fusion import (
    RuleConstrainedMLPConfig,
    build_symbolic_feature_frame,
    predict_rule_constrained_mlp,
    symbolic_posthoc_correction,
    train_learned_fusion,
    train_rule_constrained_mlp,
    uncertainty_aware_fusion,
    weighted_fusion,
)
from ..models.heuristic import legacy_heuristic_score
from ..models.sequence import (
    RecurrentPredictor,
    TransformerPredictor,
    mc_dropout_uncertainty,
    predict_torch_model,
    split_sequence_data,
    train_torch_model,
)
from ..paths import DATASETS_DIR, EXPERIMENTS_DIR, RUNTIME_DIR, ensure_directories
from ..rules.engine import FuzzyTemporalRuleEngine
from ..utils import write_json
from .metrics import bootstrap_confidence_interval, classification_metrics, paired_bootstrap_difference, regression_metrics, reliability_curve
from .robustness import inject_missingness, inject_noise


RISK_TARGETS = {"verification_failure", "overreliance_risk", "high_offloading_behavior", "target_any_failure"}
SUCCESS_TARGETS = {"delayed_recall_success", "transfer_success"}
FEATURE_GROUPS = {
    "behavioral": ["cognitive_offloading_index", "copy_paste_dependence", "edit_distance_reliance", "prompt_depth_score", "prompt_count"],
    "verification": ["verification_robustness", "source_checking_rate", "citation_support", "flawed_answer_acceptance"],
    "temporal": ["rolling_offloading", "rolling_verification", "offloading_trend", "session_index", "ai_exposure_hours"],
    "learning": ["retention_gap", "transfer_gap", "delayed_recall_score", "transfer_score"],
    "metacognitive": ["confidence", "calibration_error", "reflection_depth"],
}
RULE_GROUP_TO_DRIVER = {
    "verification": "verification",
    "retention": "retention",
    "transfer": "transfer",
    "calibration": "calibration",
    "temporal": "offloading",
    "protective": "offloading",
}


@dataclass
class ExperimentArtifacts:
    dataset_paths: dict[str, Path]
    experiment_paths: dict[str, Path]
    best_models_path: Path


def _classification_metric_row(target: str, model_name: str, y_true: np.ndarray, y_prob: np.ndarray, synthetic_label: str) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_prob)
    ci_low, ci_high = bootstrap_confidence_interval(lambda a, b: classification_metrics(a, b)["auroc"], y_true, y_prob, n_boot=120)
    row = {
        "target": target,
        "model": model_name,
        "data_label": synthetic_label,
        **metrics,
        "auroc_ci_low": ci_low,
        "auroc_ci_high": ci_high,
    }
    return row


def _regression_metric_row(target: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, synthetic_label: str) -> dict[str, Any]:
    metrics = regression_metrics(y_true, y_pred)
    ci_low, ci_high = bootstrap_confidence_interval(lambda a, b: regression_metrics(a, b)["mae"], y_true, y_pred, n_boot=120)
    return {
        "target": target,
        "model": model_name,
        "data_label": synthetic_label,
        **metrics,
        "mae_ci_low": ci_low,
        "mae_ci_high": ci_high,
    }


def _symbolic_probability(frame: pd.DataFrame, target: str) -> np.ndarray:
    raw = frame[f"symbolic::{target}"].to_numpy(dtype=float)
    if target in SUCCESS_TARGETS:
        return np.clip(1.0 - raw, 0.0, 1.0)
    return np.clip(raw, 0.0, 1.0)


def _heuristic_probability(frame: pd.DataFrame, target: str) -> np.ndarray:
    raw = legacy_heuristic_score(frame)
    if target in SUCCESS_TARGETS:
        return np.clip(1.0 - raw.to_numpy(dtype=float), 0.0, 1.0)
    return raw.to_numpy(dtype=float)


def _save_dataset_tables(dataset: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paths = {}
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in dataset.items():
        path = DATASETS_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def _evaluate_symbolic_engine(config: ProjectConfig, sessions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    engine = FuzzyTemporalRuleEngine(config)
    evaluations = [engine.evaluate_row(row) for _, row in sessions.iterrows()]
    symbolic_frame = build_symbolic_feature_frame(evaluations)
    for target in config.targets + config.regression_targets:
        symbolic_frame[f"symbolic::{target}"] = [evaluation["target_scores"].get(target, 0.5) for evaluation in evaluations]
    group_columns = [column for column in symbolic_frame.columns if column.startswith("rule_group::")]
    symbolic_frame["top_rule_group"] = (
        symbolic_frame[group_columns].abs().idxmax(axis=1).str.replace("rule_group::", "", regex=False)
        if group_columns
        else "none"
    )
    return symbolic_frame, evaluations


def run_full_experiment(config: ProjectConfig) -> ExperimentArtifacts:
    ensure_directories()
    task_catalog = build_air_bench_catalog(config)
    dataset = simulate_air_bench_dataset(config, task_catalog)
    dataset_paths = _save_dataset_tables(dataset)

    sessions = dataset["sessions"].copy().reset_index(drop=True)
    symbolic_frame, symbolic_evaluations = _evaluate_symbolic_engine(config, sessions)
    modeling_frame = pd.concat([sessions, symbolic_frame], axis=1)

    rule_activation_rows = []
    for session_id, evaluation in zip(modeling_frame["session_id"], symbolic_evaluations, strict=True):
        for activation in evaluation["activations"]:
            rule_activation_rows.append({"session_id": session_id, **activation})
    pd.DataFrame(rule_activation_rows).to_csv(EXPERIMENTS_DIR / "rule_activations.csv", index=False)
    write_json(EXPERIMENTS_DIR / "symbolic_explanations.json", symbolic_evaluations)

    classification_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    feature_ablation_rows: list[dict[str, Any]] = []
    rule_ablation_rows: list[dict[str, Any]] = []
    condition_effects_rows: list[dict[str, Any]] = []
    model_store: dict[str, Any] = {}

    for target in config.targets:
        bundle = train_classification_baselines(modeling_frame, target, config.simulation.seed)
        test_frame = modeling_frame.loc[bundle.test_indices].reset_index(drop=True)
        train_frame = modeling_frame.loc[bundle.train_indices].reset_index(drop=True)

        heuristic_prob = _heuristic_probability(test_frame, target)
        symbolic_prob = _symbolic_probability(test_frame, target)
        classification_rows.append(_classification_metric_row(target, "heuristic_baseline", bundle.y_test, heuristic_prob, "synthetic_validation"))
        classification_rows.append(_classification_metric_row(target, "symbolic_only", bundle.y_test, symbolic_prob, "synthetic_validation"))

        best_neural_name = None
        best_neural_prob = None
        best_neural_auroc = -np.inf

        for model_name, model in bundle.models.items():
            y_prob = model.predict_proba(bundle.X_test)[:, 1]
            row = _classification_metric_row(target, model_name, bundle.y_test, y_prob, "synthetic_validation")
            classification_rows.append(row)
            if np.isnan(row["auroc"]):
                continue
            if row["auroc"] > best_neural_auroc:
                best_neural_auroc = row["auroc"]
                best_neural_name = model_name
                best_neural_prob = y_prob

        seq_data = split_sequence_data(modeling_frame, target, config.model.sequence_length, config.simulation.seed, "classification")
        for seq_name, model in {
            "gru": RecurrentPredictor(len(seq_data.feature_columns), config.model.hidden_dim, config.model.dropout, "gru"),
            "lstm": RecurrentPredictor(len(seq_data.feature_columns), config.model.hidden_dim, config.model.dropout, "lstm"),
            "transformer": TransformerPredictor(len(seq_data.feature_columns), config.model.hidden_dim, config.model.dropout),
        }.items():
            trained = train_torch_model(model, seq_data)
            probs = predict_torch_model(trained, seq_data.X_test, "classification")
            classification_rows.append(_classification_metric_row(target, seq_name, seq_data.y_test, probs, "synthetic_validation"))
            if target == "overreliance_risk":
                model_store[f"sequence::{seq_name}"] = trained

        if best_neural_name is None or best_neural_prob is None:
            continue

        model_store[f"classification::{target}"] = {
            "model_name": best_neural_name,
            "feature_columns": bundle.feature_columns,
            "model": bundle.models[best_neural_name],
        }

        uncertainty = bootstrap_classification_uncertainty(
            lambda: type(bundle.models[best_neural_name])(**bundle.models[best_neural_name].get_params()),
            bundle.X_train,
            bundle.y_train,
            bundle.X_test,
            config.model.bootstrap_samples,
            config.simulation.seed,
        )
        top_rule_activation = test_frame["top_rule_activation"].to_numpy(dtype=float)
        weighted_prob = weighted_fusion(best_neural_prob, symbolic_prob)
        corrected_prob = symbolic_posthoc_correction(best_neural_prob, symbolic_prob, top_rule_activation)
        uncertainty_prob = uncertainty_aware_fusion(best_neural_prob, symbolic_prob, uncertainty)

        fusion_train = pd.DataFrame(
            {
                "neural_prob": bundle.models[best_neural_name].predict_proba(bundle.X_train)[:, 1],
                "symbolic_prob": _symbolic_probability(train_frame, target),
                "uncertainty": bootstrap_classification_uncertainty(
                    lambda: type(bundle.models[best_neural_name])(**bundle.models[best_neural_name].get_params()),
                    bundle.X_train,
                    bundle.y_train,
                    bundle.X_train,
                    min(15, config.model.bootstrap_samples),
                    config.simulation.seed,
                ),
                "top_rule_activation": train_frame["top_rule_activation"].to_numpy(dtype=float),
            }
        )
        fusion_test = pd.DataFrame(
            {
                "neural_prob": best_neural_prob,
                "symbolic_prob": symbolic_prob,
                "uncertainty": uncertainty,
                "top_rule_activation": top_rule_activation,
            }
        )
        learned_fusion = train_learned_fusion(
            pd.concat([fusion_train, train_frame[[target]].reset_index(drop=True)], axis=1),
            target,
            list(fusion_train.columns),
        )
        learned_prob = learned_fusion.predict_proba(fusion_test.to_numpy(dtype=float))[:, 1]

        rule_constrained = train_rule_constrained_mlp(
            bundle.X_train,
            bundle.y_train,
            _symbolic_probability(train_frame, target),
            RuleConstrainedMLPConfig(input_dim=bundle.X_train.shape[1], hidden_dim=config.model.hidden_dim, dropout=config.model.dropout),
        )
        constrained_prob = predict_rule_constrained_mlp(rule_constrained, bundle.X_test)

        for name, probs in {
            "weighted_fusion": weighted_prob,
            "symbolic_posthoc_correction": corrected_prob,
            "uncertainty_aware_fusion": uncertainty_prob,
            "learned_fusion": learned_prob,
            "rule_constrained_mlp": constrained_prob,
        }.items():
            classification_rows.append(_classification_metric_row(target, name, bundle.y_test, probs, "synthetic_validation"))

        if target == "overreliance_risk":
            calibration_rows.extend(reliability_curve(bundle.y_test, learned_prob))
            model_store["fusion::learned"] = learned_fusion
            model_store["fusion::rule_constrained_mlp"] = rule_constrained

            for scenario_name, transformed in {
                "missing_10": inject_missingness(bundle.X_test, 0.10, config.simulation.seed),
                "missing_30": inject_missingness(bundle.X_test, 0.30, config.simulation.seed),
                "noise_05": inject_noise(bundle.X_test, 0.05, config.simulation.seed),
                "noise_10": inject_noise(bundle.X_test, 0.10, config.simulation.seed),
            }.items():
                neural_pred = bundle.models[best_neural_name].predict_proba(transformed)[:, 1]
                fused = weighted_fusion(neural_pred, symbolic_prob)
                metrics = classification_metrics(bundle.y_test, fused)
                robustness_rows.append({"scenario": scenario_name, "model": "weighted_fusion", **metrics})

            for group_name, group_features in FEATURE_GROUPS.items():
                ablated_features = [feature for feature in bundle.feature_columns if feature not in group_features]
                ablated_model = train_classification_baselines(modeling_frame[ablated_features + [target]], target, config.simulation.seed)
                ablated_prob = ablated_model.models["calibrated_gradient_boosting"].predict_proba(ablated_model.X_test)[:, 1]
                metrics = classification_metrics(ablated_model.y_test, ablated_prob)
                feature_ablation_rows.append({"group": group_name, **metrics})

            full_symbolic = classification_metrics(bundle.y_test, symbolic_prob)
            for group_name in sorted({rule["group"] for rule in pd.DataFrame(rule_activation_rows).to_dict("records")}):
                masked = test_frame.copy()
                masked_symbolic = symbolic_prob.copy()
                affected = pd.DataFrame(symbolic_evaluations)[[]] if False else None
                zero_mask = [activation["group"] == group_name for activation in symbolic_evaluations[0]["activations"]] if symbolic_evaluations else []
                if masked_symbolic.size:
                    group_strength = test_frame.filter(like=f"rule_group::{group_name}")
                    if not group_strength.empty:
                        masked_symbolic = np.clip(symbolic_prob - group_strength.sum(axis=1).to_numpy(dtype=float), 0.0, 1.0)
                metrics = classification_metrics(bundle.y_test, masked_symbolic)
                diff = paired_bootstrap_difference(lambda a, b: classification_metrics(a, b)["auroc"], bundle.y_test, symbolic_prob, masked_symbolic, n_boot=120)
                rule_ablation_rows.append({"group": group_name, **metrics, **diff, "full_symbolic_auroc": full_symbolic["auroc"]})

    for target in config.regression_targets:
        bundle = train_regression_baselines(modeling_frame, target, config.simulation.seed)
        for model_name, model in bundle.models.items():
            predictions = model.predict(bundle.X_test)
            regression_rows.append(_regression_metric_row(target, model_name, bundle.y_test, predictions, "synthetic_validation"))
        seq_data = split_sequence_data(modeling_frame, target, config.model.sequence_length, config.simulation.seed, "regression")
        for seq_name, model in {
            "gru_regressor": RecurrentPredictor(len(seq_data.feature_columns), config.model.hidden_dim, config.model.dropout, "gru"),
            "lstm_regressor": RecurrentPredictor(len(seq_data.feature_columns), config.model.hidden_dim, config.model.dropout, "lstm"),
            "transformer_regressor": TransformerPredictor(len(seq_data.feature_columns), config.model.hidden_dim, config.model.dropout),
        }.items():
            trained = train_torch_model(model, seq_data)
            predictions = predict_torch_model(trained, seq_data.X_test, "regression")
            regression_rows.append(_regression_metric_row(target, seq_name, seq_data.y_test, predictions, "synthetic_validation"))

    for condition_id, group in modeling_frame.groupby("condition_id"):
        condition_effects_rows.append(
            {
                "condition_id": condition_id,
                "immediate_success": group["immediate_success"].mean(),
                "delayed_recall_success": group["delayed_recall_success"].mean(),
                "transfer_success": group["transfer_success"].mean(),
                "verification_failure": group["verification_failure"].mean(),
                "overreliance_risk": group["overreliance_risk"].mean(),
                "intervention_benefit": group["intervention_benefit"].mean(),
                "data_label": "synthetic_validation",
            }
        )

    mapped_rule_driver = modeling_frame["top_rule_group"].map(RULE_GROUP_TO_DRIVER).fillna(modeling_frame["top_rule_group"])
    explanation_faithfulness = float((mapped_rule_driver == modeling_frame["latent_driver"]).mean())
    sample_idx = int(modeling_frame["overreliance_score"].idxmax())
    sample_report = {
        "session": modeling_frame.loc[sample_idx].to_dict(),
        "symbolic_explanation": symbolic_evaluations[sample_idx]["explanation"],
        "note": "Synthetic participant session report for demonstration only.",
    }

    classification_df = pd.DataFrame(classification_rows)
    regression_df = pd.DataFrame(regression_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    robustness_df = pd.DataFrame(robustness_rows)
    feature_ablation_df = pd.DataFrame(feature_ablation_rows)
    rule_ablation_df = pd.DataFrame(rule_ablation_rows)
    condition_effects_df = pd.DataFrame(condition_effects_rows)

    classification_df.to_csv(EXPERIMENTS_DIR / "classification_results.csv", index=False)
    regression_df.to_csv(EXPERIMENTS_DIR / "regression_results.csv", index=False)
    calibration_df.to_csv(EXPERIMENTS_DIR / "calibration_curve.csv", index=False)
    robustness_df.to_csv(EXPERIMENTS_DIR / "robustness_results.csv", index=False)
    feature_ablation_df.to_csv(EXPERIMENTS_DIR / "feature_ablation_results.csv", index=False)
    rule_ablation_df.to_csv(EXPERIMENTS_DIR / "rule_ablation_results.csv", index=False)
    condition_effects_df.to_csv(EXPERIMENTS_DIR / "condition_effects.csv", index=False)
    write_json(EXPERIMENTS_DIR / "sample_session_report.json", sample_report)

    macro_results = (
        classification_df.groupby("model")[["auroc", "auprc", "f1", "balanced_accuracy", "ece", "brier_score"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("auroc", ascending=False)
    )
    macro_results.to_csv(EXPERIMENTS_DIR / "classification_macro_results.csv", index=False)

    summary = {
        "synthetic_validation_only": True,
        "n_users": int(config.simulation.n_users),
        "sessions_per_user": int(config.simulation.sessions_per_user),
        "total_sessions": int(len(modeling_frame)),
        "task_families": sorted(modeling_frame["task_family"].unique().tolist()),
        "conditions": sorted(modeling_frame["condition_id"].unique().tolist()),
        "explanation_faithfulness": explanation_faithfulness,
        "best_classification_model": macro_results.iloc[0].to_dict() if not macro_results.empty else {},
        "note": "All metrics in this summary are from synthetic or simulated data and are pending empirical validation on real human-subject data.",
    }
    write_json(EXPERIMENTS_DIR / "experiment_summary.json", summary)

    best_models_path = RUNTIME_DIR / "best_models.pkl"
    with best_models_path.open("wb") as handle:
        pickle.dump(model_store, handle)

    return ExperimentArtifacts(
        dataset_paths=dataset_paths,
        experiment_paths={
            "classification_results": EXPERIMENTS_DIR / "classification_results.csv",
            "regression_results": EXPERIMENTS_DIR / "regression_results.csv",
            "calibration_curve": EXPERIMENTS_DIR / "calibration_curve.csv",
            "robustness_results": EXPERIMENTS_DIR / "robustness_results.csv",
            "feature_ablation_results": EXPERIMENTS_DIR / "feature_ablation_results.csv",
            "rule_ablation_results": EXPERIMENTS_DIR / "rule_ablation_results.csv",
            "condition_effects": EXPERIMENTS_DIR / "condition_effects.csv",
            "macro_results": EXPERIMENTS_DIR / "classification_macro_results.csv",
            "summary": EXPERIMENTS_DIR / "experiment_summary.json",
            "sample_session_report": EXPERIMENTS_DIR / "sample_session_report.json",
        },
        best_models_path=best_models_path,
    )
