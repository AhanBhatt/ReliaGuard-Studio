from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.adapters.registry import get_dataset_registry
from ..paths import REAL_DATA_EXPERIMENTS_DIR


def build_construct_support_matrix() -> pd.DataFrame:
    constructs = [
        "appropriate reliance",
        "overreliance",
        "underreliance",
        "confidence shift",
        "learning gain",
        "process traces",
        "verification proxy",
        "metacognitive engagement",
        "outcome performance",
        "delayed recall",
        "transfer",
    ]
    rows = []
    for key, entry in get_dataset_registry().items():
        supported = set(entry.card.supported_constructs)
        row = {"dataset_key": key, "dataset_name": entry.card.name, "decision": entry.card.decision}
        for construct in constructs:
            row[construct] = int(construct in supported)
        rows.append(row)
    return pd.DataFrame(rows)


def run_cross_dataset_summary(prepared_interactions: dict[str, pd.DataFrame], model_results: pd.DataFrame) -> dict[str, Path]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    construct_matrix = build_construct_support_matrix()
    construct_path = REAL_DATA_EXPERIMENTS_DIR / "construct_matrix.csv"
    construct_matrix.to_csv(construct_path, index=False)

    summary_rows = []
    for dataset_name, interactions in prepared_interactions.items():
        row = {
            "dataset_name": dataset_name,
            "n_interactions": int(len(interactions)),
            "n_participants": int(interactions["participant_id"].nunique()) if "participant_id" in interactions.columns else 0,
            "initial_accuracy": float(interactions["initial_correct"].mean()) if "initial_correct" in interactions.columns else float("nan"),
            "final_accuracy": float(interactions["final_correct"].mean()) if "final_correct" in interactions.columns else float("nan"),
            "overreliance_rate": float(interactions["overreliance"].mean()) if "overreliance" in interactions.columns else float("nan"),
            "underreliance_rate": float(interactions["underreliance"].mean()) if "underreliance" in interactions.columns else float("nan"),
            "appropriate_reliance_rate": float(interactions["appropriate_reliance"].mean()) if "appropriate_reliance" in interactions.columns else float("nan"),
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    model_summary = (
        model_results.groupby(["dataset", "target", "model"])[["auroc", "ece", "balanced_accuracy"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["dataset", "target", "auroc"], ascending=[True, True, False])
    )
    model_summary_path = REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_model_summary.csv"
    model_summary.to_csv(model_summary_path, index=False)
    return {
        "construct_matrix": construct_path,
        "cross_dataset_summary": summary_path,
        "cross_dataset_model_summary": model_summary_path,
    }
