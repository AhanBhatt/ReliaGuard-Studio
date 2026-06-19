from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR
from .metrics import bootstrap_confidence_interval, classification_metrics


HARMONIZED_FEATURES = [
    "initial_correct",
    "advice_correct",
    "initial_confidence",
    "confidence_change",
    "stated_accuracy_normalized",
    "post_explain_reliability",
    "advice_source_ai",
    "xai_present",
    "conversational_xai",
    "user_question_rate",
    "task_position_normalized",
]


def _load_interactions() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for dataset_dir in REAL_DATA_PREPARED_DIR.iterdir():
        path = dataset_dir / "interactions.csv"
        if path.exists():
            frames[dataset_dir.name] = pd.read_csv(path, low_memory=False)
    return frames


def _harmonize(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    output = frame.copy()
    for column in HARMONIZED_FEATURES:
        if column not in output.columns:
            output[column] = 0.0
    if "stated_accuracy_normalized" in output.columns and "post_explain_reliability" in output.columns:
        output["stated_accuracy_normalized"] = output["stated_accuracy_normalized"].fillna(output["post_explain_reliability"])
        output["post_explain_reliability"] = output["post_explain_reliability"].fillna(output["stated_accuracy_normalized"])
    output["initial_confidence"] = output["initial_confidence"].fillna(0.5)
    output["confidence_change"] = output["confidence_change"].fillna(0.0)
    output["task_position_normalized"] = output["task_position_normalized"].fillna(0.5)
    output = output.dropna(subset=[target])
    return output


def _target_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "overreliance":
        return frame.loc[(frame["initial_correct"] == 1) & (frame["advice_correct"] == 0)].copy()
    if target == "underreliance":
        return frame.loc[(frame["initial_correct"] == 0) & (frame["advice_correct"] == 1)].copy()
    if target == "appropriate_reliance":
        return frame.loc[frame["disagreement_case"] == 1].copy()
    raise ValueError(target)


def _evaluate_transfer(train: pd.DataFrame, test: pd.DataFrame, target: str, model_name: str, seed: int) -> dict[str, Any]:
    if len(train) < 30 or len(test) < 30 or train[target].nunique() < 2 or test[target].nunique() < 2:
        return {"model": model_name, "n_train": len(train), "n_test": len(test), "auroc": np.nan, "ece": np.nan, "note": "insufficient class variation"}
    X_train = train[HARMONIZED_FEATURES].astype(float).fillna(0.0)
    X_test = test[HARMONIZED_FEATURES].astype(float).fillna(0.0)
    y_train = train[target].astype(int).to_numpy()
    y_test = test[target].astype(int).to_numpy()
    if model_name == "harmonized_logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    else:
        model = HistGradientBoostingClassifier(max_depth=3, random_state=seed)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, probs)
    low, high = bootstrap_confidence_interval(lambda a, b: classification_metrics(a, b)["auroc"], y_test, probs, n_boot=80, seed=seed)
    return {
        "model": model_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        **metrics,
        "auroc_ci_low": low,
        "auroc_ci_high": high,
        "note": "Harmonized-feature external transfer; dataset-specific features intentionally excluded.",
    }


def run_cross_dataset_generalization(interactions_map: dict[str, pd.DataFrame] | None = None, seed: int = 42) -> dict[str, Path]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    interactions_map = interactions_map or _load_interactions()
    datasets = [name for name in ["haiid", "convxai_iui2025", "chi2023_dke"] if name in interactions_map]
    targets = ["appropriate_reliance", "overreliance", "underreliance"]
    rows: list[dict[str, Any]] = []
    for target in targets:
        prepared = {
            name: _harmonize(_target_frame(frame, target), target)
            for name, frame in interactions_map.items()
            if name in datasets and target in frame.columns
        }
        for train_name, train_frame in prepared.items():
            for test_name, test_frame in prepared.items():
                if train_name == test_name:
                    continue
                for model_name in ["harmonized_logistic", "harmonized_gradient_boosting"]:
                    rows.append(
                        {
                            "target": target,
                            "train_dataset": train_name,
                            "test_dataset": test_name,
                            **_evaluate_transfer(train_frame, test_frame, target, model_name, seed),
                        }
                    )
        if len(prepared) >= 3:
            for heldout in prepared:
                train_frame = pd.concat([frame for name, frame in prepared.items() if name != heldout], ignore_index=True)
                test_frame = prepared[heldout]
                for model_name in ["harmonized_logistic", "harmonized_gradient_boosting"]:
                    rows.append(
                        {
                            "target": target,
                            "train_dataset": "all_except_" + heldout,
                            "test_dataset": heldout,
                            **_evaluate_transfer(train_frame, test_frame, target, model_name, seed),
                        }
                    )
    results = pd.DataFrame(rows)
    path = REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_results.csv"
    results.to_csv(path, index=False)
    return {"cross_dataset_results": path}
