from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR


def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    if y_true.nunique(dropna=True) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    if y_true.nunique(dropna=True) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _label_permutation_controls(rng: np.random.Generator) -> list[dict[str, object]]:
    predictions_path = REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv"
    if not predictions_path.exists():
        return []
    pred = pd.read_csv(predictions_path)
    rows: list[dict[str, object]] = []
    for (dataset, target, split, model), group in pred.groupby(["dataset", "target", "split", "model"]):
        if len(group) < 30 or group["y_true"].nunique() < 2:
            continue
        permuted = group["y_true"].to_numpy().copy()
        rng.shuffle(permuted)
        rows.append(
            {
                "control": "label_permutation",
                "dataset": dataset,
                "target": target,
                "split": split,
                "model": model,
                "n": int(len(group)),
                "prevalence": float(group["y_true"].mean()),
                "auroc_original": _safe_auc(group["y_true"], group["y_prob"]),
                "auroc_control": _safe_auc(pd.Series(permuted), group["y_prob"]),
                "auprc_original": _safe_auprc(group["y_true"], group["y_prob"]),
                "auprc_control": _safe_auprc(pd.Series(permuted), group["y_prob"]),
                "interpretation": "Predictive signal should collapse toward prevalence/chance after label permutation.",
            }
        )
    return rows


def _advice_shuffle_controls(rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_dir in sorted(REAL_DATA_PREPARED_DIR.glob("*")):
        interactions_path = dataset_dir / "interactions.csv"
        if not interactions_path.exists():
            continue
        df = pd.read_csv(interactions_path, low_memory=False)
        required = {"initial_correct", "final_correct", "advice_correct"}
        if not required.issubset(df.columns):
            continue
        applicable = df.dropna(subset=list(required)).copy()
        if applicable.empty:
            continue
        shuffled = applicable["advice_correct"].to_numpy().copy()
        if "task_family" in applicable.columns:
            for _, idx in applicable.groupby("task_family").groups.items():
                idx_array = np.asarray(list(idx))
                shuffled[idx_array] = rng.permutation(shuffled[idx_array])
        else:
            rng.shuffle(shuffled)
        initial = applicable["initial_correct"].astype(int).to_numpy()
        final = applicable["final_correct"].astype(int).to_numpy()
        original_advice = applicable["advice_correct"].astype(int).to_numpy()
        original_over = ((initial == 1) & (original_advice == 0) & (final == 0)).mean()
        original_under = ((initial == 0) & (original_advice == 1) & (final == 0)).mean()
        shuffled_over = ((initial == 1) & (shuffled == 0) & (final == 0)).mean()
        shuffled_under = ((initial == 0) & (shuffled == 1) & (final == 0)).mean()
        rows.extend(
            [
                {
                    "control": "advice_correctness_shuffle",
                    "dataset": dataset_dir.name,
                    "target": "overreliance_proxy",
                    "split": "within_task_family_shuffle",
                    "model": "state_definition",
                    "n": int(len(applicable)),
                    "prevalence": float(original_over),
                    "auroc_original": None,
                    "auroc_control": None,
                    "auprc_original": None,
                    "auprc_control": None,
                    "control_rate": float(shuffled_over),
                    "interpretation": "State rates are recomputed after shuffling advice correctness within task strata.",
                },
                {
                    "control": "advice_correctness_shuffle",
                    "dataset": dataset_dir.name,
                    "target": "underreliance_proxy",
                    "split": "within_task_family_shuffle",
                    "model": "state_definition",
                    "n": int(len(applicable)),
                    "prevalence": float(original_under),
                    "auroc_original": None,
                    "auroc_control": None,
                    "auprc_original": None,
                    "auprc_control": None,
                    "control_rate": float(shuffled_under),
                    "interpretation": "State rates are recomputed after shuffling advice correctness within task strata.",
                },
            ]
        )
    return rows


def run_negative_controls(seed: int = 2026) -> dict[str, Path]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = _label_permutation_controls(rng)
    rows.extend(_advice_shuffle_controls(rng))
    df = pd.DataFrame(rows)
    out = REAL_DATA_EXPERIMENTS_DIR / "negative_controls.csv"
    df.to_csv(out, index=False)
    return {"negative_controls": out}


if __name__ == "__main__":
    run_negative_controls()
