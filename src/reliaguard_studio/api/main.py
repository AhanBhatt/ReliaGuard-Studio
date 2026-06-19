from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config.loader import load_project_config
from .product import (
    RelianceCase,
    ablation_results,
    conformal_thresholds,
    dataset_summary,
    evaluation_lab,
    explain_case,
    model_card,
    policy_simulation,
    predict_reliance_case,
)
from ..data.storage import init_db, save_report, save_session
from ..evaluation.runner import run_full_experiment
from ..models.heuristic import legacy_heuristic_score
from ..paths import DATASETS_DIR, EXPERIMENTS_DIR, PACKAGE_ROOT, RUNTIME_DIR
from ..rules.engine import FuzzyTemporalRuleEngine


class SessionScoreRequest(BaseModel):
    session_id: str | None = None
    payload: dict[str, Any] | None = None


app = FastAPI(
    title="ReliaGuard Studio API",
    version="0.2.0",
    description="Production-style API for reliance-state prediction, explanation, conformal gating, and model-evaluation dashboards.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PACKAGE_ROOT / "frontend" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _ensure_artifacts() -> None:
    config = load_project_config()
    if not (DATASETS_DIR / "sessions.csv").exists() or not (RUNTIME_DIR / "best_models.pkl").exists():
        run_full_experiment(config)
    init_db()


def _load_sessions() -> pd.DataFrame:
    _ensure_artifacts()
    return pd.read_csv(DATASETS_DIR / "sessions.csv")


def _load_tasks() -> pd.DataFrame:
    _ensure_artifacts()
    return pd.read_csv(DATASETS_DIR / "tasks.csv")


def _load_model_store() -> dict[str, Any]:
    _ensure_artifacts()
    with (RUNTIME_DIR / "best_models.pkl").open("rb") as handle:
        return pickle.load(handle)


def _score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = load_project_config()
    engine = FuzzyTemporalRuleEngine(config)
    symbolic = engine.evaluate_row(payload)
    heuristic = float(legacy_heuristic_score(pd.DataFrame([payload])).iloc[0])
    model_store = _load_model_store()

    over_model = model_store.get("classification::overreliance_risk")
    if over_model:
        feature_columns = over_model["feature_columns"]
        model = over_model["model"]
        feature_frame = pd.DataFrame([payload]).reindex(columns=feature_columns, fill_value=0.0)
        neural = float(model.predict_proba(feature_frame.to_numpy(dtype=float))[:, 1][0])
    else:
        neural = heuristic

    symbolic_over = float(symbolic["target_scores"].get("overreliance_risk", 0.5))
    fusion = float(max(0.0, min(1.0, 0.65 * neural + 0.35 * symbolic_over)))
    uncertainty = float(abs(neural - symbolic_over))
    report = {
        "neural_overreliance_probability": neural,
        "symbolic_overreliance_probability": symbolic_over,
        "fusion_overreliance_probability": fusion,
        "uncertainty": uncertainty,
        "symbolic": symbolic,
        "heuristic_risk_score": heuristic,
        "privacy_notice": "This local-first supplementary demo stores only session reports written on this machine.",
        "synthetic_validation_notice": "This interactive demo uses AIR-Bench stress-test sessions. The manuscript's main empirical evidence comes from public real decision-making datasets, not from this demo.",
    }
    return report


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ReliaGuard Studio"}


@app.post("/predict-reliance")
@app.post("/api/predict-reliance")
def predict_reliance(case: RelianceCase) -> dict[str, Any]:
    return predict_reliance_case(case).model_dump()


@app.post("/explain-case")
@app.post("/api/explain-case")
def explain_reliance_case(case: RelianceCase) -> dict[str, Any]:
    return explain_case(case)


@app.get("/conformal-threshold")
@app.get("/api/conformal-threshold")
def conformal_threshold(alpha: float = 0.10) -> dict[str, Any]:
    thresholds = conformal_thresholds(alpha=alpha)
    source = thresholds[0].get("alpha_source", "unavailable") if thresholds else "unavailable"
    served_alpha = thresholds[0].get("served_artifact_alpha") if thresholds else None
    return {
        "alpha": alpha,
        "served_artifact_alpha": served_alpha,
        "alpha_source": source,
        "thresholds": thresholds,
        "boundary": (
            "Exact rows are calibrated artifacts. Preview rows are monotone UI estimates from the nearest "
            "artifact alpha and should be regenerated before scientific reporting."
        ),
    }


@app.get("/simulate-policy")
@app.get("/api/simulate-policy")
def simulate_policy() -> dict[str, Any]:
    return {"policies": policy_simulation(), "boundary": "Observational simulation only; not a causal deployment effect."}


@app.get("/run-ablation")
@app.get("/api/run-ablation")
def run_ablation() -> dict[str, Any]:
    return {"ablation_results": ablation_results()}


@app.get("/datasets")
@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    return {"datasets": dataset_summary()}


@app.get("/model-card")
@app.get("/api/model-card")
def relia_model_card() -> dict[str, Any]:
    return model_card()


@app.get("/evaluation-lab")
@app.get("/api/evaluation-lab")
def model_evaluation_lab() -> dict[str, Any]:
    return evaluation_lab()


@app.get("/api/config")
def config() -> dict[str, Any]:
    return load_project_config().model_dump()


@app.get("/api/dashboard/summary")
def dashboard_summary() -> dict[str, Any]:
    _ensure_artifacts()
    summary = pd.read_json(EXPERIMENTS_DIR / "experiment_summary.json", typ="series").to_dict()
    condition_effects = pd.read_csv(EXPERIMENTS_DIR / "condition_effects.csv").to_dict(orient="records")
    return {"summary": summary, "condition_effects": condition_effects}


@app.get("/api/demo/sample-session")
def sample_session(task_family: str | None = None, condition_id: str | None = None) -> dict[str, Any]:
    sessions = _load_sessions()
    tasks = _load_tasks()
    if task_family:
        sessions = sessions.loc[sessions["task_family"] == task_family]
    if condition_id:
        sessions = sessions.loc[sessions["condition_id"] == condition_id]
    if sessions.empty:
        raise HTTPException(status_code=404, detail="No matching AIR-Bench stress-test session found.")
    session = sessions.sample(1, random_state=42).iloc[0].to_dict()
    task = tasks.loc[tasks["task_id"] == session["task_id"]].iloc[0].to_dict()
    return {"session": session, "task": task}


@app.post("/api/score")
def score(request: SessionScoreRequest) -> dict[str, Any]:
    sessions = _load_sessions()
    if request.session_id:
        match = sessions.loc[sessions["session_id"] == request.session_id]
        if match.empty:
            raise HTTPException(status_code=404, detail="Session not found.")
        payload = match.iloc[0].to_dict()
    elif request.payload:
        payload = request.payload
    else:
        raise HTTPException(status_code=400, detail="Either session_id or payload must be provided.")
    report = _score_payload(payload)
    session_id = str(payload.get("session_id", "ad_hoc_session"))
    try:
        save_session(
            session_id=session_id,
            user_id=str(payload.get("user_id", "demo")),
            condition_id=str(payload.get("condition_id", "unknown")),
            task_family=str(payload.get("task_family", "unknown")),
            payload=payload,
        )
        save_report(session_id, report)
    except OSError as exc:
        report["storage_warning"] = f"Runtime persistence skipped: {exc}"
    except Exception as exc:
        if "readonly" in str(exc).lower() or "permission" in str(exc).lower():
            report["storage_warning"] = f"Runtime persistence skipped: {exc}"
        else:
            raise
    return report
