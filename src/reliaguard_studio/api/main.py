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
from .platform import (
    GuardrailCheck,
    IngestRequest,
    InteractionEvent,
    ProjectCreate,
    ReplayRequest,
    ReviewLabel,
    create_project,
    demo_customer_support_records,
    get_project,
    guardrail_check,
    ingest_records,
    intervention_templates,
    label_review,
    list_projects,
    log_interaction,
    monitoring_summary,
    preview_columns,
    reports as platform_reports,
    replay_logs,
    review_queue,
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


@app.get("/v1/projects")
@app.get("/api/v1/projects")
def api_projects() -> dict[str, Any]:
    return {"projects": list_projects()}


@app.post("/v1/projects")
@app.post("/api/v1/projects")
def api_create_project(payload: ProjectCreate) -> dict[str, Any]:
    return create_project(payload)


@app.get("/v1/projects/{project_id}")
@app.get("/api/v1/projects/{project_id}")
def api_project(project_id: str) -> dict[str, Any]:
    project = get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.post("/v1/events/log")
@app.post("/api/v1/events/log")
def api_log_interaction(payload: InteractionEvent) -> dict[str, Any]:
    return log_interaction(payload)


@app.post("/v1/guardrail/check")
@app.post("/api/v1/guardrail/check")
def api_guardrail_check(payload: GuardrailCheck) -> dict[str, Any]:
    return guardrail_check(payload)


@app.post("/v1/ingest/preview")
@app.post("/api/v1/ingest/preview")
def api_ingest_preview(records: list[dict[str, Any]]) -> dict[str, Any]:
    return preview_columns(records)


@app.post("/v1/ingest/validate")
@app.post("/api/v1/ingest/validate")
def api_ingest_validate(payload: IngestRequest) -> dict[str, Any]:
    return ingest_records(payload)


@app.get("/v1/review-queue")
@app.get("/api/v1/review-queue")
def api_review_queue(project_id: str | None = None) -> dict[str, Any]:
    return {"cases": review_queue(project_id)}


@app.post("/v1/review-queue/label")
@app.post("/api/v1/review-queue/label")
def api_label_review(payload: ReviewLabel) -> dict[str, Any]:
    try:
        return label_review(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/interventions")
@app.get("/api/v1/interventions")
def api_interventions() -> dict[str, Any]:
    return {"interventions": intervention_templates()}


@app.get("/v1/reports")
@app.get("/api/v1/reports")
def api_reports(project_id: str | None = None) -> dict[str, Any]:
    return {"reports": platform_reports(project_id)}


@app.post("/v1/replay")
@app.post("/api/v1/replay")
def api_replay(payload: ReplayRequest) -> dict[str, Any]:
    return replay_logs(payload)


@app.get("/v1/monitoring")
@app.get("/api/v1/monitoring")
def api_monitoring(project_id: str | None = None) -> dict[str, Any]:
    return monitoring_summary(project_id)


@app.get("/v1/demo/customer-support")
@app.get("/api/v1/demo/customer-support")
def api_demo_customer_support() -> dict[str, Any]:
    return {
        "vertical": "Customer Support Copilot",
        "description": "Support agents choose refund, deny refund, escalate, or request more information; ReliaGuard audits over-acceptance and under-use of AI advice.",
        "records": demo_customer_support_records(),
        "mapping": {
            "user_id": "agent_id",
            "task_id": "ticket_id",
            "initial_answer": "human_first_decision",
            "initial_confidence": "confidence_before_ai",
            "ai_advice": "model_recommendation",
            "ai_confidence": "model_confidence",
            "final_answer": "human_final_decision",
            "ground_truth": "true_label",
            "context": "domain",
        },
    }


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
