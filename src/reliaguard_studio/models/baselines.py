from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor


IDENTIFIER_COLUMNS = {
    "user_id",
    "session_id",
    "task_id",
    "task_family",
    "condition_id",
    "condition_name",
    "education_level",
    "latent_driver",
    "condition_group",
}


TARGET_COLUMNS = {
    "delayed_recall_success",
    "transfer_success",
    "verification_failure",
    "overreliance_risk",
    "high_offloading_behavior",
    "retention_gap",
    "target_any_failure",
}


@dataclass
class BaselineBundle:
    task_type: Literal["classification", "regression"]
    feature_columns: list[str]
    train_indices: np.ndarray
    test_indices: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    models: dict[str, object]


def get_feature_columns(frame: pd.DataFrame, extra_exclusions: set[str] | None = None) -> list[str]:
    exclusions = set(TARGET_COLUMNS) | IDENTIFIER_COLUMNS | (extra_exclusions or set())
    numeric_cols = frame.select_dtypes(include=["number", "bool"]).columns
    return [column for column in numeric_cols if column not in exclusions]


def build_split(
    frame: pd.DataFrame, target: str, seed: int, test_size: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    feature_columns = get_feature_columns(frame, {target})
    X = frame[feature_columns].to_numpy(dtype=float)
    y = frame[target].to_numpy()
    indices = frame.index.to_numpy()
    stratify = y if set(np.unique(y)).issubset({0, 1}) else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=test_size, random_state=seed, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, feature_columns, idx_train, idx_test


def train_classification_baselines(frame: pd.DataFrame, target: str, seed: int) -> BaselineBundle:
    X_train, X_test, y_train, y_test, feature_columns, idx_train, idx_test = build_split(frame, target, seed)
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1200, random_state=seed, solver="liblinear"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced"),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=seed, max_depth=4),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
    }
    models["calibrated_gradient_boosting"] = CalibratedClassifierCV(
        HistGradientBoostingClassifier(random_state=seed, max_depth=4), cv=3, method="sigmoid"
    )
    for model in models.values():
        model.fit(X_train, y_train)
    return BaselineBundle(
        task_type="classification",
        feature_columns=feature_columns,
        train_indices=idx_train,
        test_indices=idx_test,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models=models,
    )


def train_regression_baselines(frame: pd.DataFrame, target: str, seed: int) -> BaselineBundle:
    X_train, X_test, y_train, y_test, feature_columns, idx_train, idx_test = build_split(frame, target, seed)
    models: dict[str, object] = {
        "ridge": Ridge(random_state=seed),
        "random_forest_regressor": RandomForestRegressor(n_estimators=200, random_state=seed),
        "gradient_boosting_regressor": HistGradientBoostingRegressor(random_state=seed, max_depth=4),
        "mlp_regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return BaselineBundle(
        task_type="regression",
        feature_columns=feature_columns,
        train_indices=idx_train,
        test_indices=idx_test,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models=models,
    )


def bootstrap_classification_uncertainty(model_factory, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(n_boot):
        indices = rng.choice(len(X_train), size=len(X_train), replace=True)
        model = model_factory()
        model.fit(X_train[indices], y_train[indices])
        predictions.append(model.predict_proba(X_test)[:, 1])
    return np.std(np.vstack(predictions), axis=0)


def bootstrap_regression_uncertainty(model_factory, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(n_boot):
        indices = rng.choice(len(X_train), size=len(X_train), replace=True)
        model = model_factory()
        model.fit(X_train[indices], y_train[indices])
        predictions.append(model.predict(X_test))
    return np.std(np.vstack(predictions), axis=0)


def baseline_counterfactual_delta(model: object, feature_columns: list[str], sample: pd.Series, feature: str, delta: float) -> float:
    altered = sample.copy()
    altered[feature] = np.clip(altered[feature] + delta, 0.0, 1.0)
    baseline = sample[feature_columns].to_numpy(dtype=float).reshape(1, -1)
    modified = altered[feature_columns].to_numpy(dtype=float).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(modified)[:, 1] - model.predict_proba(baseline)[:, 1])
    return float(mean_absolute_error(model.predict(modified), model.predict(baseline)))
