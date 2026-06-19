from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REPO_ROOT


def log_experiment_summary_to_mlflow(tracking_uri: str | None = None) -> dict[str, str]:
    """Log headline model metrics to MLflow when the optional dependency is installed.

    The function is intentionally optional so the public repository remains lightweight.
    Install with `python -m pip install -e .[mlops]` to enable experiment tracking.
    """

    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is absent
        raise RuntimeError("MLflow is optional. Install with `python -m pip install -e .[mlops]`.") from exc

    tracking_uri = tracking_uri or str(REPO_ROOT / "artifacts" / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("reliaguard-studio")
    model_results = REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv"
    if not model_results.exists():
        raise FileNotFoundError(model_results)
    frame = pd.read_csv(model_results)
    with mlflow.start_run(run_name="public-real-data-summary"):
        for metric in ["auroc", "auprc", "f1", "balanced_accuracy", "brier_score", "ece"]:
            if metric in frame.columns:
                mlflow.log_metric(f"median_{metric}", float(frame[metric].median()))
        mlflow.log_artifact(str(model_results))
    return {"tracking_uri": tracking_uri}
