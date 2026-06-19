from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.adapters.registry import get_dataset_registry, get_integrated_adapters, get_manual_only_adapters
from ..models.baselines import bootstrap_classification_uncertainty
from ..models.fusion import train_learned_fusion, uncertainty_aware_fusion, weighted_fusion
from ..paths import REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR, REAL_DATA_REPORTS_DIR, ensure_directories
from ..rules.engine import FuzzyTemporalRuleEngine
from ..rules.real_data import RealRuleBundle, build_chi2023_rule_bundle, build_convxai_rule_bundle, build_haiid_rule_bundle
from ..utils import write_json
from .calibration_analysis import build_calibration_summary
from .cross_dataset import run_cross_dataset_summary
from .cross_dataset_generalization import run_cross_dataset_generalization
from .gee_models import run_gee_models
from .metrics import bootstrap_confidence_interval, classification_metrics, paired_bootstrap_difference, regression_metrics, reliability_curve
from .mixed_effects import mixed_effects_status_table
from .policy_evaluation import run_policy_evaluation
from .sensitivity import write_sensitivity_outputs
from .statistics import cohens_d, summarize_rate


@dataclass
class RealExperimentArtifacts:
    prepared_paths: dict[str, dict[str, Path]]
    experiment_paths: dict[str, Path]


def _prepared_path_index(dataset_names: list[str] | None = None) -> dict[str, dict[str, Path]]:
    names = dataset_names or sorted(path.name for path in REAL_DATA_PREPARED_DIR.iterdir() if path.is_dir())
    return {
        name: {
            "interactions": REAL_DATA_PREPARED_DIR / name / "interactions.csv",
            "participants": REAL_DATA_PREPARED_DIR / name / "participants.csv",
            "tasks": REAL_DATA_PREPARED_DIR / name / "tasks.csv",
            "metadata": REAL_DATA_PREPARED_DIR / name / "metadata.json",
        }
        for name in names
    }


def _experiment_path_index() -> dict[str, Path]:
    return {
        "haiid_descriptive_metrics": REAL_DATA_EXPERIMENTS_DIR / "haiid_descriptive_metrics.csv",
        "chi2023_condition_effects": REAL_DATA_EXPERIMENTS_DIR / "chi2023_condition_effects.csv",
        "chi2023_miscalibration_effects": REAL_DATA_EXPERIMENTS_DIR / "chi2023_miscalibration_effects.csv",
        "convxai_condition_effects": REAL_DATA_EXPERIMENTS_DIR / "convxai_condition_effects.csv",
        "pardos_learning_effects": REAL_DATA_EXPERIMENTS_DIR / "pardos_learning_effects.csv",
        "flora_process_effects": REAL_DATA_EXPERIMENTS_DIR / "flora_process_effects.csv",
        "flora_process_models": REAL_DATA_EXPERIMENTS_DIR / "flora_process_models.csv",
        "state_distribution": REAL_DATA_EXPERIMENTS_DIR / "reliance_state_distribution.csv",
        "model_results": REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv",
        "calibration_curves": REAL_DATA_EXPERIMENTS_DIR / "real_calibration_curves.csv",
        "rule_ablation_results": REAL_DATA_EXPERIMENTS_DIR / "real_rule_ablation_results.csv",
        "error_analysis": REAL_DATA_EXPERIMENTS_DIR / "real_error_analysis.csv",
        "rule_activations": REAL_DATA_EXPERIMENTS_DIR / "real_rule_activations.csv",
        "predictions": REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv",
        "effect_sizes": REAL_DATA_EXPERIMENTS_DIR / "effect_sizes.csv",
        "split_robustness": REAL_DATA_EXPERIMENTS_DIR / "split_robustness.csv",
        "ablation_summary": REAL_DATA_EXPERIMENTS_DIR / "ablation_summary.csv",
        "clustered_condition_contrasts": REAL_DATA_EXPERIMENTS_DIR / "clustered_condition_contrasts.csv",
        "gee_results": REAL_DATA_EXPERIMENTS_DIR / "gee_results.csv",
        "mixed_effects_status": REAL_DATA_EXPERIMENTS_DIR / "mixed_effects_status.csv",
        "cross_dataset_summary": REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_summary.csv",
        "cross_dataset_model_summary": REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_model_summary.csv",
        "cross_dataset_results": REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_results.csv",
        "policy_evaluation": REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv",
        "policy_actions": REAL_DATA_EXPERIMENTS_DIR / "policy_actions.csv",
        "policy_burden_sensitivity": REAL_DATA_EXPERIMENTS_DIR / "policy_burden_sensitivity.csv",
        "sensitivity_summary": REAL_DATA_EXPERIMENTS_DIR / "sensitivity_summary.csv",
        "decision_curve_summary": REAL_DATA_EXPERIMENTS_DIR / "decision_curve_summary.csv",
        "label_definition_sensitivity": REAL_DATA_EXPERIMENTS_DIR / "label_definition_sensitivity.csv",
        "calibration_summary": REAL_DATA_EXPERIMENTS_DIR / "calibration_summary.csv",
        "calibration_curves_detailed": REAL_DATA_EXPERIMENTS_DIR / "calibration_curves_detailed.csv",
        "statistical_summary": REAL_DATA_EXPERIMENTS_DIR / "statistical_summary.json",
        "summary": REAL_DATA_EXPERIMENTS_DIR / "real_experiment_summary.json",
        "report": REAL_DATA_REPORTS_DIR / "real_data_notes.md",
    }


def cached_real_experiment_artifacts() -> RealExperimentArtifacts | None:
    prepared_paths = _prepared_path_index()
    experiment_paths = _experiment_path_index()
    required_prepared = [paths["interactions"] for paths in prepared_paths.values()]
    required_experiments = [experiment_paths[key] for key in ["model_results", "summary", "statistical_summary", "gee_results", "cross_dataset_results", "policy_evaluation", "calibration_summary"]]
    if required_prepared and all(path.exists() for path in required_prepared) and all(path.exists() for path in required_experiments):
        return RealExperimentArtifacts(prepared_paths=prepared_paths, experiment_paths=experiment_paths)
    return None


def download_real_data(force: bool = False) -> dict[str, list[Path]]:
    ensure_directories()
    outputs: dict[str, list[Path]] = {}
    for key, entry in get_dataset_registry().items():
        outputs[key] = entry.adapter.download(force=force)
    return outputs


def prepare_real_data(force_download: bool = False) -> dict[str, Any]:
    ensure_directories()
    prepared: dict[str, Any] = {}
    for key, adapter in get_integrated_adapters().items():
        if force_download:
            adapter.download(force=True)
        prepared[key] = adapter.prepare()
    return prepared


def _build_symbolic_feature_frame(rule_evaluations: list[dict[str, Any]], targets: list[str]) -> pd.DataFrame:
    rows = []
    for evaluation in rule_evaluations:
        row: dict[str, float] = {"top_rule_activation": 0.0}
        for target in targets:
            row[f"symbolic::{target}"] = float(evaluation["target_scores"].get(target, 0.5))
        grouped: dict[str, float] = {}
        for activation in evaluation["activations"]:
            key = f"rule_group::{activation['group']}"
            grouped[key] = grouped.get(key, 0.0) + float(activation["activation"]) * float(activation["signed_weight"])
            row["top_rule_activation"] = max(row["top_rule_activation"], float(activation["activation"]))
        row.update(grouped)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def _evaluate_rule_bundle(frame: pd.DataFrame, bundle: RealRuleBundle) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    engine = FuzzyTemporalRuleEngine(SimpleNamespace(targets=bundle.targets, regression_targets=bundle.regression_targets, rules=bundle.rules))
    evaluations = [engine.evaluate_row(row) for _, row in frame.iterrows()]
    symbolic_frame = _build_symbolic_feature_frame(evaluations, bundle.targets)
    return symbolic_frame, evaluations


def _classification_row(dataset: str, target: str, split: str, model: str, y_true: np.ndarray, y_prob: np.ndarray, seed: int) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_prob)
    low, high = bootstrap_confidence_interval(lambda a, b: classification_metrics(a, b)["auroc"], y_true, y_prob, n_boot=80, seed=seed)
    return {
        "dataset": dataset,
        "target": target,
        "split": split,
        "model": model,
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        **metrics,
        "auroc_ci_low": low,
        "auroc_ci_high": high,
    }


def _make_splits(frame: pd.DataFrame, target: str, split_kind: str, group_col: str | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col and group_col in frame.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, test_idx = next(splitter.split(frame, frame[target], groups=frame[group_col]))
    else:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(frame))
        rng.shuffle(indices)
        cutoff = max(1, int(0.75 * len(frame)))
        train_idx, test_idx = indices[:cutoff], indices[cutoff:]
    return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)


def _encode_features(train_frame: pd.DataFrame, test_frame: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    combined = pd.concat([train_frame[feature_columns], test_frame[feature_columns]], axis=0, ignore_index=True)
    encoded = pd.get_dummies(combined, columns=[column for column in feature_columns if combined[column].dtype == "object"], dummy_na=True)
    train_encoded = encoded.iloc[: len(train_frame)].copy()
    test_encoded = encoded.iloc[len(train_frame) :].copy()
    fill_values = train_encoded.median(numeric_only=True)
    train_encoded = train_encoded.fillna(fill_values).fillna(0.0)
    test_encoded = test_encoded.fillna(fill_values).fillna(0.0)
    return train_encoded.to_numpy(dtype=float), test_encoded.to_numpy(dtype=float), train_encoded.columns.tolist()


def _model_factories(seed: int) -> dict[str, tuple[Any, Any]]:
    return {
        "majority_baseline": (
            DummyClassifier(strategy="prior", random_state=seed),
            lambda: DummyClassifier(strategy="prior", random_state=seed),
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=1200, solver="liblinear", class_weight="balanced", random_state=seed),
            lambda: LogisticRegression(max_iter=1200, solver="liblinear", class_weight="balanced", random_state=seed),
        ),
        "calibrated_logistic_regression": (
            CalibratedClassifierCV(
                LogisticRegression(max_iter=1200, solver="liblinear", class_weight="balanced", random_state=seed),
                cv=3,
                method="sigmoid",
            ),
            lambda: LogisticRegression(max_iter=1200, solver="liblinear", class_weight="balanced", random_state=seed),
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=150, random_state=seed, class_weight="balanced"),
            lambda: RandomForestClassifier(n_estimators=150, random_state=seed, class_weight="balanced"),
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(random_state=seed, max_depth=4),
            lambda: HistGradientBoostingClassifier(random_state=seed, max_depth=4),
        ),
        "calibrated_gradient_boosting": (
            CalibratedClassifierCV(HistGradientBoostingClassifier(random_state=seed, max_depth=4), cv=3, method="sigmoid"),
            lambda: HistGradientBoostingClassifier(random_state=seed, max_depth=4),
        ),
        "mlp": (
            MLPClassifier(hidden_layer_sizes=(24,), max_iter=250, early_stopping=True, random_state=seed),
            lambda: MLPClassifier(hidden_layer_sizes=(24,), max_iter=250, early_stopping=True, random_state=seed),
        ),
    }


def _run_binary_task(
    frame: pd.DataFrame,
    dataset: str,
    target: str,
    feature_columns: list[str],
    symbolic_column: str,
    split_name: str,
    group_col: str | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    if frame[target].nunique() < 2 or len(frame) < 40:
        return [], [], [], [], pd.DataFrame()

    train_frame, test_frame = _make_splits(frame, target, split_name, group_col, seed)
    X_train, X_test, encoded_names = _encode_features(train_frame, test_frame, feature_columns)
    y_train = train_frame[target].to_numpy(dtype=int)
    y_test = test_frame[target].to_numpy(dtype=int)
    symbolic_train = train_frame[symbolic_column].to_numpy(dtype=float)
    symbolic_test = test_frame[symbolic_column].to_numpy(dtype=float)
    top_rule_train = train_frame["top_rule_activation"].to_numpy(dtype=float)
    top_rule_test = test_frame["top_rule_activation"].to_numpy(dtype=float)

    result_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    rule_ablation_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    result_rows.append(_classification_row(dataset, target, split_name, "symbolic_only", y_test, symbolic_test, seed))

    best_neural_name = ""
    best_neural_prob: np.ndarray | None = None
    best_neural_auroc = -np.inf
    best_factory = None
    for model_name, (model, factory) in _model_factories(seed).items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        row = _classification_row(dataset, target, split_name, model_name, y_test, probs, seed)
        result_rows.append(row)
        if np.isfinite(row["auroc"]) and row["auroc"] > best_neural_auroc:
            best_neural_auroc = row["auroc"]
            best_neural_name = model_name
            best_neural_prob = probs
            best_factory = factory

    if best_neural_prob is None or best_factory is None:
        return result_rows, calibration_rows, rule_ablation_rows, error_rows, pd.DataFrame()

    best_train_model = best_factory()
    best_train_model.fit(X_train, y_train)
    best_train_prob = best_train_model.predict_proba(X_train)[:, 1]
    uncertainty = bootstrap_classification_uncertainty(best_factory, X_train, y_train, X_test, n_boot=6, seed=seed)
    uncertainty_train = bootstrap_classification_uncertainty(best_factory, X_train, y_train, X_train, n_boot=4, seed=seed)
    weighted_prob = weighted_fusion(best_neural_prob, symbolic_test)
    uncertainty_prob = uncertainty_aware_fusion(best_neural_prob, symbolic_test, uncertainty)
    fusion_train = pd.DataFrame(
        {
            "neural_prob": best_train_prob,
            "symbolic_prob": symbolic_train,
            "uncertainty": uncertainty_train,
            "top_rule_activation": top_rule_train,
            target: y_train,
        }
    )
    learned_fusion = train_learned_fusion(fusion_train, target, ["neural_prob", "symbolic_prob", "uncertainty", "top_rule_activation"])
    learned_prob = learned_fusion.predict_proba(
        pd.DataFrame(
            {
                "neural_prob": best_neural_prob,
                "symbolic_prob": symbolic_test,
                "uncertainty": uncertainty,
                "top_rule_activation": top_rule_test,
            }
        ).to_numpy(dtype=float)
    )[:, 1]
    reliance_state_prob = np.clip(
        0.40 * learned_prob
        + 0.30 * uncertainty_prob
        + 0.20 * symbolic_test
        + 0.10 * np.clip(top_rule_test, 0.0, 1.0),
        0.0,
        1.0,
    )

    fusion_predictions: list[pd.DataFrame] = []
    for model_name, probs in {
        "weighted_fusion": weighted_prob,
        "uncertainty_aware_fusion": uncertainty_prob,
        "learned_fusion": learned_prob,
        "reliance_state_neurosymbolic": reliance_state_prob,
    }.items():
        result_rows.append(_classification_row(dataset, target, split_name, model_name, y_test, probs, seed))
        for curve_row in reliability_curve(y_test, probs):
            calibration_rows.append({"dataset": dataset, "target": target, "split": split_name, "model": model_name, **curve_row})
        fusion_predictions.append(
            pd.DataFrame(
                {
                    "participant_id": test_frame["participant_id"].to_numpy(),
                    "task_instance_key": test_frame["task_instance_key"].to_numpy(),
                    "y_true": y_test,
                    "y_prob": probs,
                    "dataset": dataset,
                    "target": target,
                    "split": split_name,
                    "model": model_name,
                }
            )
        )

    group_columns = [column for column in frame.columns if column.startswith("rule_group::")]
    for group_column in group_columns:
        masked_symbolic = np.clip(symbolic_test - test_frame[group_column].to_numpy(dtype=float), 0.0, 1.0)
        metrics = _classification_row(dataset, target, split_name, f"ablation::{group_column}", y_test, masked_symbolic, seed)
        diff = paired_bootstrap_difference(lambda a, b: classification_metrics(a, b)["auroc"], y_test, symbolic_test, masked_symbolic, n_boot=80, seed=seed)
        rule_ablation_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "split": split_name,
                "group": group_column.replace("rule_group::", ""),
                **metrics,
                **diff,
            }
        )

    ranked = pd.DataFrame(
        {
            "participant_id": test_frame["participant_id"],
            "task_instance_key": test_frame["task_instance_key"],
            "y_true": y_test,
            "y_prob": learned_prob,
            "dataset": dataset,
            "target": target,
            "split": split_name,
        }
    )
    ranked["error"] = np.abs(ranked["y_true"] - ranked["y_prob"])
    errors = pd.concat(
        [
            ranked.loc[ranked["y_true"] == 1].sort_values("y_prob").head(5).assign(error_type="false_negative_risk"),
            ranked.loc[ranked["y_true"] == 0].sort_values("y_prob", ascending=False).head(5).assign(error_type="false_positive_risk"),
        ],
        ignore_index=True,
    )
    error_rows.extend(errors.to_dict("records"))

    prediction_frame = pd.concat(fusion_predictions, ignore_index=True) if fusion_predictions else pd.DataFrame()
    return result_rows, calibration_rows, rule_ablation_rows, error_rows, prediction_frame


def _haiid_feature_columns() -> list[str]:
    return [
        "task_family",
        "advice_source_label",
        "stated_accuracy_normalized",
        "initial_response_value",
        "initial_confidence",
        "advice_value",
        "advice_correct",
        "age_numeric",
        "education_level_numeric",
        "programming_experience_numeric",
        "socioeconomic_status_numeric",
        "expert_years",
        "task_family_overreliance_rate",
    ]


def _chi_feature_columns() -> list[str]:
    return [
        "condition_id",
        "tutorial_present",
        "xai_present",
        "initial_correct",
        "advice_correct",
        "task_position_normalized",
        "ati_scale",
        "propensity_to_trust",
        "trust_first",
        "first_batch_self_assessment_gap",
        "first_batch_overestimation",
        "first_batch_underestimation",
    ]


def _convxai_feature_columns() -> list[str]:
    return [
        "condition_id",
        "xai_present",
        "conversational_xai",
        "llm_agent",
        "initial_correct",
        "advice_correct",
        "initial_confidence",
        "confidence_change",
        "post_explain_reliability",
        "task_position_normalized",
        "user_question_count",
        "user_question_rate",
        "chat_turn_count",
        "journey_step_count",
        "pdp_count",
        "shap_count",
        "whatif_count",
        "counterfactual_count",
        "decision_tree_count",
    ]


def _describe_haiid(interactions: pd.DataFrame) -> pd.DataFrame:
    by_family = summarize_rate(interactions, ["task_family", "advice_source_label"], "overreliance", cluster_col="participant_id")
    by_family["metric"] = "overreliance"
    rows = [by_family]
    for metric in ["underreliance", "correct_ai_reliance", "correct_self_reliance", "advice_uptake", "final_correct"]:
        summary = summarize_rate(interactions, ["task_family", "advice_source_label"], metric, cluster_col="participant_id")
        summary["metric"] = metric
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _describe_chi(interactions: pd.DataFrame, participants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = interactions.loc[(interactions["analysis_included"] == 1) & (interactions["analysis_batch"] == "second")].copy()
    condition_effects = summarize_rate(analysis, ["condition_name"], "appropriate_reliance", cluster_col="participant_id")
    condition_effects["metric"] = "appropriate_reliance"
    accuracy_effects = summarize_rate(analysis, ["condition_name"], "final_correct", cluster_col="participant_id")
    accuracy_effects["metric"] = "final_correct"
    overreliance = summarize_rate(analysis, ["condition_name"], "overreliance", cluster_col="participant_id")
    overreliance["metric"] = "overreliance"
    miscalibration_effects = (
        analysis.groupby("miscalibration_group")[["appropriate_reliance", "final_correct", "overreliance", "underreliance"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    miscalibration_effects["n_participants"] = participants.groupby("first_batch_miscalibration_group").size().reindex(miscalibration_effects["miscalibration_group"]).values
    return pd.concat([condition_effects, accuracy_effects, overreliance], ignore_index=True), miscalibration_effects


def _describe_convxai(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["appropriate_reliance", "final_correct", "overreliance", "underreliance", "advice_uptake"]:
        summary = summarize_rate(interactions.loc[interactions["disagreement_case"] == 1], ["condition_name"], metric, cluster_col="participant_id")
        summary["metric"] = metric
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _describe_pardos(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, subframe in interactions.groupby("condition_name"):
        gain = subframe["learning_gain"].to_numpy(dtype=float)
        low, high = bootstrap_confidence_interval(lambda a, b: float(np.mean(a)), gain, gain, n_boot=200, seed=42)
        rows.append(
            {
                "condition_name": condition,
                "n": int(len(subframe)),
                "pre_test_score": float(subframe["pre_test_score"].mean()),
                "post_test_score": float(subframe["post_test_score"].mean()),
                "learning_gain": float(subframe["learning_gain"].mean()),
                "ci_low": low,
                "ci_high": high,
                "session_time_seconds": float(subframe["session_time_seconds"].mean()),
                "total_unique_steps": float(subframe["total_unique_steps"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("condition_name")


def _two_group_mean_difference(values: pd.Series, groups: pd.Series, n_boot: int = 200, seed: int = 42) -> tuple[float, float, float]:
    high = values.loc[groups == 1].dropna().to_numpy(dtype=float)
    low = values.loc[groups == 0].dropna().to_numpy(dtype=float)
    if len(high) == 0 or len(low) == 0:
        return float("nan"), float("nan"), float("nan")
    effect = float(np.mean(high) - np.mean(low))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        high_sample = rng.choice(high, size=len(high), replace=True)
        low_sample = rng.choice(low, size=len(low), replace=True)
        estimates.append(float(np.mean(high_sample) - np.mean(low_sample)))
    return effect, float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _describe_flora(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "genai_intensity",
        "prompt_depth",
        "metacognitive_prompt_ratio",
        "evaluation_prompt_ratio",
        "source_engagement_proxy",
        "revision_depth",
        "copy_paste_write_ratio",
        "process_diversity",
    ]
    for metric in metrics:
        if metric not in interactions.columns:
            continue
        median_group = (interactions[metric] >= interactions[metric].median()).astype(int)
        effect, low, high = _two_group_mean_difference(interactions["proposal_score_normalized"], median_group, n_boot=180, seed=42)
        rows.append(
            {
                "process_feature": metric,
                "n": int(interactions[metric].notna().sum()),
                "mean_feature_value": float(interactions[metric].mean()),
                "high_feature_mean_score": float(interactions.loc[median_group == 1, "proposal_score_normalized"].mean()),
                "low_feature_mean_score": float(interactions.loc[median_group == 0, "proposal_score_normalized"].mean()),
                "score_difference_high_minus_low": effect,
                "ci_low": low,
                "ci_high": high,
                "analysis_note": "observational median-split process association; not causal",
            }
        )
    return pd.DataFrame(rows)


def _run_flora_process_models(interactions: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    feature_columns = [
        "prompt_count",
        "prompt_depth",
        "prompt_code_diversity",
        "metacognitive_prompt_ratio",
        "evaluation_prompt_ratio",
        "planning_prompt_ratio",
        "chatbot_critique_ratio",
        "writing_event_count",
        "final_word_count",
        "revision_depth",
        "copy_paste_write_ratio",
        "annotation_count",
        "source_engagement_proxy",
        "process_diversity",
        "prior_ds_knowledge",
        "prior_genai_knowledge",
        "genai_familiarity",
        "temporal_regulation_profile",
    ]
    available = [column for column in feature_columns if column in interactions.columns]
    frame = interactions.dropna(subset=["proposal_score_normalized"]).copy()
    if len(frame) < 40:
        return pd.DataFrame()
    X = pd.get_dummies(frame[available], dummy_na=True).fillna(0.0)
    y = frame["proposal_score_normalized"].to_numpy(dtype=float)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "ridge": make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=4, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15, random_state=seed),
    }
    rows = []
    for name, model in models.items():
        predictions = cross_val_predict(model, X, y, cv=cv)
        metrics = regression_metrics(y, predictions)
        low, high = bootstrap_confidence_interval(lambda a, b: regression_metrics(a, b)["r2"], y, predictions, n_boot=140, seed=seed)
        rows.append(
            {
                "dataset": "flora_ips",
                "target": "proposal_score_normalized",
                "split": "student_level_5fold",
                "model": name,
                "n": int(len(frame)),
                **metrics,
                "r2_ci_low": low,
                "r2_ci_high": high,
                "analysis_note": "observational process-trace prediction; one row per student",
            }
        )
    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def _build_effect_size_table(interactions_map: dict[str, pd.DataFrame], seed: int = 42) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in interactions_map.items():
        if dataset_name == "pardos_chatgpt_tutoring" and "learning_gain" in frame.columns:
            gain = frame["learning_gain"].to_numpy(dtype=float)
            low, high = bootstrap_confidence_interval(lambda a, b: float(np.mean(a)), gain, gain, n_boot=160, seed=seed)
            rows.append(
                {
                    "dataset": dataset_name,
                    "contrast": "post-test minus pre-test score",
                    "n_participants": int(frame["participant_id"].nunique()),
                    "effect": float(np.mean(gain)),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
            continue
        if dataset_name == "flora_ips" and {"proposal_score_normalized", "metacognitive_prompt_ratio"}.issubset(frame.columns):
            median_group = (frame["metacognitive_prompt_ratio"] >= frame["metacognitive_prompt_ratio"].median()).astype(int)
            effect, low, high = _two_group_mean_difference(frame["proposal_score_normalized"], median_group, n_boot=160, seed=seed)
            rows.append(
                {
                    "dataset": dataset_name,
                    "contrast": "high versus low metacognitive prompt ratio score",
                    "n_participants": int(frame["participant_id"].nunique()),
                    "effect": effect,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
            continue
        if {"initial_correct", "final_correct", "participant_id"}.issubset(frame.columns):
            participant_delta = (
                frame.groupby("participant_id")[["initial_correct", "final_correct"]]
                .mean(numeric_only=True)
                .assign(delta=lambda df: df["final_correct"] - df["initial_correct"])
            )
            low, high = bootstrap_confidence_interval(
                lambda a, b: float(np.mean(a)),
                participant_delta["delta"].to_numpy(dtype=float),
                participant_delta["delta"].to_numpy(dtype=float),
                n_boot=80,
                seed=seed,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "contrast": "final minus initial accuracy",
                    "n_participants": int(len(participant_delta)),
                    "effect": float(participant_delta["delta"].mean()),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _build_split_robustness_table(model_results: pd.DataFrame) -> pd.DataFrame:
    if model_results.empty:
        return pd.DataFrame()
    preferred_models = ["reliance_state_neurosymbolic", "uncertainty_aware_fusion", "weighted_fusion", "calibrated_gradient_boosting"]
    filtered = model_results.loc[model_results["model"].isin(preferred_models)].copy()
    return (
        filtered.sort_values(["dataset", "target", "split", "auroc", "ece"], ascending=[True, True, True, False, True])
        .groupby(["dataset", "target", "split"])
        .head(1)
        .reset_index(drop=True)
    )


def _build_ablation_summary(rule_ablation: pd.DataFrame) -> pd.DataFrame:
    if rule_ablation.empty:
        return pd.DataFrame()
    return (
        rule_ablation.groupby(["dataset", "target", "group"])
        .agg(
            mean_auroc_delta=("mean_difference", "mean"),
            ci_low=("ci_low", "mean"),
            ci_high=("ci_high", "mean"),
            n_splits=("split", "nunique"),
        )
        .reset_index()
        .sort_values(["dataset", "target", "mean_auroc_delta"])
    )


def _build_clustered_condition_table(interactions_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in interactions_map.items():
        if "condition_name" not in frame.columns:
            continue
        analysis = frame.loc[frame["disagreement_case"] == 1].copy()
        if analysis.empty:
            continue
        baseline = sorted(analysis["condition_name"].dropna().unique())[0]
        base_rate = float(analysis.loc[analysis["condition_name"] == baseline, "appropriate_reliance"].mean())
        for condition, subframe in analysis.groupby("condition_name"):
            rate = float(subframe["appropriate_reliance"].mean())
            low, high = summarize_rate(subframe, ["condition_name"], "appropriate_reliance", cluster_col="participant_id")[["ci_low", "ci_high"]].iloc[0]
            rows.append(
                {
                    "dataset": dataset_name,
                    "reference_condition": baseline,
                    "condition": condition,
                    "n": int(len(subframe)),
                    "appropriate_reliance": rate,
                    "difference_from_reference": rate - base_rate,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "model_note": "participant-cluster bootstrap contrast; not a fitted mixed-effects logistic model",
                }
            )
    return pd.DataFrame(rows)


def run_real_experiments(seed: int = 42, force: bool = False) -> RealExperimentArtifacts:
    ensure_directories()
    if not force:
        cached = cached_real_experiment_artifacts()
        if cached is not None:
            return cached
    prepared_datasets = prepare_real_data(force_download=False)
    prepared_paths = _prepared_path_index(sorted(prepared_datasets))
    interactions_map = {name: dataset.interactions.copy() for name, dataset in prepared_datasets.items()}
    participants_map = {name: dataset.participants.copy() for name, dataset in prepared_datasets.items()}

    descriptive_haiid = _describe_haiid(interactions_map["haiid"])
    chi_condition_effects, chi_miscalibration = _describe_chi(interactions_map["chi2023_dke"], participants_map["chi2023_dke"])
    convxai_condition_effects = _describe_convxai(interactions_map["convxai_iui2025"])
    pardos_learning_effects = (
        _describe_pardos(interactions_map["pardos_chatgpt_tutoring"])
        if "pardos_chatgpt_tutoring" in interactions_map
        else pd.DataFrame()
    )
    flora_process_effects = (
        _describe_flora(interactions_map["flora_ips"])
        if "flora_ips" in interactions_map
        else pd.DataFrame()
    )
    flora_process_models = (
        _run_flora_process_models(interactions_map["flora_ips"], seed=seed)
        if "flora_ips" in interactions_map
        else pd.DataFrame()
    )
    state_distribution = (
        pd.concat([frame[["dataset_name", "reliance_state"]] for frame in interactions_map.values()], ignore_index=True)
        .groupby(["dataset_name", "reliance_state"])
        .size()
        .reset_index(name="count")
    )

    results_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    rule_ablation_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    rule_activation_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    bundles = {"haiid": build_haiid_rule_bundle(), "chi2023_dke": build_chi2023_rule_bundle(), "convxai_iui2025": build_convxai_rule_bundle()}
    task_specs = [
        ("haiid", "overreliance", interactions_map["haiid"].loc[(interactions_map["haiid"]["initial_correct"] == 1) & (interactions_map["haiid"]["advice_correct"] == 0)].copy(), _haiid_feature_columns(), "participant_id", "task_family"),
        ("haiid", "underreliance", interactions_map["haiid"].loc[(interactions_map["haiid"]["initial_correct"] == 0) & (interactions_map["haiid"]["advice_correct"] == 1)].copy(), _haiid_feature_columns(), "participant_id", "task_instance_key"),
        ("chi2023_dke", "appropriate_reliance", interactions_map["chi2023_dke"].loc[(interactions_map["chi2023_dke"]["analysis_included"] == 1) & (interactions_map["chi2023_dke"]["analysis_batch"] == "second") & (interactions_map["chi2023_dke"]["disagreement_case"] == 1)].copy(), _chi_feature_columns(), "participant_id", "task_instance_key"),
        ("convxai_iui2025", "appropriate_reliance", interactions_map["convxai_iui2025"].loc[interactions_map["convxai_iui2025"]["disagreement_case"] == 1].copy(), _convxai_feature_columns(), "participant_id", "task_instance_key"),
        ("convxai_iui2025", "overreliance", interactions_map["convxai_iui2025"].loc[(interactions_map["convxai_iui2025"]["initial_correct"] == 1) & (interactions_map["convxai_iui2025"]["advice_correct"] == 0)].copy(), _convxai_feature_columns(), "participant_id", "task_instance_key"),
    ]

    for dataset_name, target, frame, feature_columns, participant_group, task_group in task_specs:
        bundle = bundles[dataset_name]
        symbolic_frame, evaluations = _evaluate_rule_bundle(frame, bundle)
        frame = pd.concat([frame.reset_index(drop=True), symbolic_frame.reset_index(drop=True)], axis=1)
        for session_row, evaluation in zip(frame[["participant_id", "task_instance_key"]].to_dict("records"), evaluations, strict=True):
            for activation in evaluation["activations"]:
                rule_activation_rows.append(
                    {
                        "dataset": dataset_name,
                        "participant_id": session_row["participant_id"],
                        "task_instance_key": session_row["task_instance_key"],
                        **activation,
                    }
                )

        for split_name, group_col in [("random", None), ("participant", participant_group), ("strict_task", task_group)]:
            task_results, task_calibration, task_ablations, task_errors, task_predictions = _run_binary_task(
                frame=frame,
                dataset=dataset_name,
                target=target,
                feature_columns=feature_columns,
                symbolic_column=f"symbolic::{target}" if f"symbolic::{target}" in frame.columns else f"symbolic::{bundle.targets[0]}",
                split_name=split_name,
                group_col=group_col,
                seed=seed,
            )
            results_rows.extend(task_results)
            calibration_rows.extend(task_calibration)
            rule_ablation_rows.extend(task_ablations)
            error_rows.extend(task_errors)
            if not task_predictions.empty:
                prediction_frames.append(task_predictions)

    model_results = pd.DataFrame(results_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    rule_ablation_df = pd.DataFrame(rule_ablation_rows)
    error_df = pd.DataFrame(error_rows)
    rule_activation_df = pd.DataFrame(rule_activation_rows)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    effect_sizes = _build_effect_size_table(interactions_map, seed=seed)
    split_robustness = _build_split_robustness_table(model_results)
    ablation_summary = _build_ablation_summary(rule_ablation_df)
    clustered_condition = _build_clustered_condition_table(interactions_map)
    gee_results = run_gee_models(interactions_map)
    mixed_effects_status = mixed_effects_status_table()

    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    descriptive_haiid_path = REAL_DATA_EXPERIMENTS_DIR / "haiid_descriptive_metrics.csv"
    chi_condition_path = REAL_DATA_EXPERIMENTS_DIR / "chi2023_condition_effects.csv"
    chi_miscalibration_path = REAL_DATA_EXPERIMENTS_DIR / "chi2023_miscalibration_effects.csv"
    convxai_condition_path = REAL_DATA_EXPERIMENTS_DIR / "convxai_condition_effects.csv"
    pardos_learning_path = REAL_DATA_EXPERIMENTS_DIR / "pardos_learning_effects.csv"
    flora_process_path = REAL_DATA_EXPERIMENTS_DIR / "flora_process_effects.csv"
    flora_process_models_path = REAL_DATA_EXPERIMENTS_DIR / "flora_process_models.csv"
    state_distribution_path = REAL_DATA_EXPERIMENTS_DIR / "reliance_state_distribution.csv"
    model_results_path = REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv"
    calibration_path = REAL_DATA_EXPERIMENTS_DIR / "real_calibration_curves.csv"
    rule_ablation_path = REAL_DATA_EXPERIMENTS_DIR / "real_rule_ablation_results.csv"
    error_path = REAL_DATA_EXPERIMENTS_DIR / "real_error_analysis.csv"
    rule_activation_path = REAL_DATA_EXPERIMENTS_DIR / "real_rule_activations.csv"
    prediction_path = REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv"
    effect_sizes_path = REAL_DATA_EXPERIMENTS_DIR / "effect_sizes.csv"
    split_robustness_path = REAL_DATA_EXPERIMENTS_DIR / "split_robustness.csv"
    ablation_summary_path = REAL_DATA_EXPERIMENTS_DIR / "ablation_summary.csv"
    clustered_condition_path = REAL_DATA_EXPERIMENTS_DIR / "clustered_condition_contrasts.csv"
    gee_results_path = REAL_DATA_EXPERIMENTS_DIR / "gee_results.csv"
    mixed_effects_status_path = REAL_DATA_EXPERIMENTS_DIR / "mixed_effects_status.csv"

    descriptive_haiid.to_csv(descriptive_haiid_path, index=False)
    chi_condition_effects.to_csv(chi_condition_path, index=False)
    chi_miscalibration.to_csv(chi_miscalibration_path, index=False)
    convxai_condition_effects.to_csv(convxai_condition_path, index=False)
    pardos_learning_effects.to_csv(pardos_learning_path, index=False)
    flora_process_effects.to_csv(flora_process_path, index=False)
    flora_process_models.to_csv(flora_process_models_path, index=False)
    state_distribution.to_csv(state_distribution_path, index=False)
    model_results.to_csv(model_results_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    rule_ablation_df.to_csv(rule_ablation_path, index=False)
    error_df.to_csv(error_path, index=False)
    rule_activation_df.to_csv(rule_activation_path, index=False)
    prediction_df.to_csv(prediction_path, index=False)
    effect_sizes.to_csv(effect_sizes_path, index=False)
    split_robustness.to_csv(split_robustness_path, index=False)
    ablation_summary.to_csv(ablation_summary_path, index=False)
    clustered_condition.to_csv(clustered_condition_path, index=False)
    gee_results.to_csv(gee_results_path, index=False)
    mixed_effects_status.to_csv(mixed_effects_status_path, index=False)

    cross_paths = run_cross_dataset_summary(interactions_map, model_results)
    cross_generalization_paths = run_cross_dataset_generalization(interactions_map, seed=seed)
    policy_paths = run_policy_evaluation(interactions_map)
    policy_frame = pd.read_csv(policy_paths["policy_evaluation"]) if "policy_evaluation" in policy_paths else pd.DataFrame()
    sensitivity_paths = write_sensitivity_outputs(interactions_map, policy_frame)
    calibration_paths = build_calibration_summary(prediction_df)

    over_group = participants_map["chi2023_dke"].loc[participants_map["chi2023_dke"]["first_batch_miscalibration_group"] == "overestimation", "second_batch_correct"].to_numpy(dtype=float)
    under_group = participants_map["chi2023_dke"].loc[participants_map["chi2023_dke"]["first_batch_miscalibration_group"] == "underestimation", "second_batch_correct"].to_numpy(dtype=float)
    dke_effect = cohens_d(over_group, under_group)

    best_rows = (
        model_results.sort_values(["dataset", "target", "auroc", "ece"], ascending=[True, True, False, True])
        .groupby(["dataset", "target"])
        .head(1)
        .reset_index(drop=True)
    )
    summary = {
        "integrated_datasets": sorted(prepared_datasets.keys()),
        "manual_only_datasets": sorted(get_manual_only_adapters().keys()),
        "proposed_title": "ReliaGuard Studio: Reliance-State Modeling and Conformal Gating Across Human-AI Environments",
        "dataset_sizes": {
            name: {
                "n_interactions": int(len(dataset.interactions)),
                "n_participants": int(dataset.interactions["participant_id"].nunique()),
            }
            for name, dataset in prepared_datasets.items()
        },
        "haiid_summary": {
            "initial_accuracy": float(interactions_map["haiid"]["initial_correct"].mean()),
            "final_accuracy": float(interactions_map["haiid"]["final_correct"].mean()),
            "overreliance_rate": float(interactions_map["haiid"]["overreliance"].mean()),
            "underreliance_rate": float(interactions_map["haiid"]["underreliance"].mean()),
        },
        "chi2023_summary": {
            "second_batch_appropriate_reliance": float(
                interactions_map["chi2023_dke"].loc[
                    (interactions_map["chi2023_dke"]["analysis_batch"] == "second")
                    & (interactions_map["chi2023_dke"]["disagreement_case"] == 1),
                    "appropriate_reliance",
                ].mean()
            ),
            "dke_second_batch_cohens_d": dke_effect,
        },
        "convxai_summary": {
            "initial_accuracy": float(interactions_map["convxai_iui2025"]["initial_correct"].mean()),
            "final_accuracy": float(interactions_map["convxai_iui2025"]["final_correct"].mean()),
            "overreliance_rate": float(interactions_map["convxai_iui2025"]["overreliance"].mean()),
            "underreliance_rate": float(interactions_map["convxai_iui2025"]["underreliance"].mean()),
            "appropriate_reliance": float(
                interactions_map["convxai_iui2025"].loc[
                    interactions_map["convxai_iui2025"]["disagreement_case"] == 1, "appropriate_reliance"
                ].mean()
            ),
        },
        "pardos_summary": (
            {
                "n_participants": int(interactions_map["pardos_chatgpt_tutoring"]["participant_id"].nunique()),
                "mean_learning_gain": float(interactions_map["pardos_chatgpt_tutoring"]["learning_gain"].mean()),
                "condition_learning_gains": pardos_learning_effects.set_index("condition_name")["learning_gain"].to_dict(),
            }
            if "pardos_chatgpt_tutoring" in interactions_map
            else {}
        ),
        "flora_summary": (
            {
                "n_students": int(interactions_map["flora_ips"]["participant_id"].nunique()),
                "mean_proposal_score": float(interactions_map["flora_ips"]["proposal_score"].mean()),
                "mean_proposal_score_normalized": float(interactions_map["flora_ips"]["proposal_score_normalized"].mean()),
                "mean_prompt_count": float(interactions_map["flora_ips"]["prompt_count"].mean()),
                "mean_metacognitive_prompt_ratio": float(interactions_map["flora_ips"]["metacognitive_prompt_ratio"].mean()),
                "best_process_model": (
                    flora_process_models.head(1).to_dict("records")[0]
                    if not flora_process_models.empty
                    else {}
                ),
            }
            if "flora_ips" in interactions_map
            else {}
        ),
        "best_models": best_rows.to_dict("records"),
        "note": (
            "Main empirical results combine three public real decision-making datasets, one public tutoring learning-gain dataset, "
            "and one public GenAI process-trace dataset. No integrated dataset supports delayed recall or transfer claims."
        ),
    }
    statistical_summary = {
        "effect_sizes": effect_sizes.to_dict("records"),
        "split_robustness": split_robustness.to_dict("records"),
        "ablation_summary": ablation_summary.to_dict("records"),
        "clustered_condition_contrasts": clustered_condition.to_dict("records"),
        "gee_results": gee_results.to_dict("records"),
        "mixed_effects_status": mixed_effects_status.to_dict("records"),
        "sensitivity_outputs": sensitivity_paths,
    }
    summary_path = REAL_DATA_EXPERIMENTS_DIR / "real_experiment_summary.json"
    statistical_summary_path = REAL_DATA_EXPERIMENTS_DIR / "statistical_summary.json"
    write_json(summary_path, summary)
    write_json(statistical_summary_path, statistical_summary)

    report_lines = [
        "# Real-data experiment notes",
        "",
        "Integrated datasets:",
        "",
        *[f"- {name}" for name in sorted(prepared_datasets.keys())],
        "",
        f"- Proposed title: {summary['proposed_title']}",
        "",
        "Main caution:",
        "",
        "- Pardos/Bhandari supports learning-gain claims, but no integrated dataset supports delayed recall, transfer, or long-term learning claims.",
        "- FLoRA supports observational GenAI process-trace and proposal-score claims, not causal claims.",
    ]
    report_path = REAL_DATA_REPORTS_DIR / "real_data_notes.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    experiment_paths = {
        "haiid_descriptive_metrics": descriptive_haiid_path,
        "chi2023_condition_effects": chi_condition_path,
        "chi2023_miscalibration_effects": chi_miscalibration_path,
        "convxai_condition_effects": convxai_condition_path,
        "pardos_learning_effects": pardos_learning_path,
        "flora_process_effects": flora_process_path,
        "flora_process_models": flora_process_models_path,
        "state_distribution": state_distribution_path,
        "model_results": model_results_path,
        "calibration_curves": calibration_path,
        "rule_ablation_results": rule_ablation_path,
        "error_analysis": error_path,
        "rule_activations": rule_activation_path,
        "predictions": prediction_path,
        "effect_sizes": effect_sizes_path,
        "split_robustness": split_robustness_path,
        "ablation_summary": ablation_summary_path,
        "clustered_condition_contrasts": clustered_condition_path,
        "gee_results": gee_results_path,
        "mixed_effects_status": mixed_effects_status_path,
        "statistical_summary": statistical_summary_path,
        "summary": summary_path,
        "report": report_path,
        **cross_paths,
        **cross_generalization_paths,
        **policy_paths,
        **{key: Path(path) for key, path in sensitivity_paths.items()},
        **calibration_paths,
    }
    return RealExperimentArtifacts(prepared_paths=prepared_paths, experiment_paths=experiment_paths)
