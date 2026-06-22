from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field

from ..paths import ARTIFACTS_DIR
from .product import RelianceCase, ReliancePrediction, predict_reliance_case


PRODUCT_DIR = ARTIFACTS_DIR / "product"
PROJECTS_FILE = PRODUCT_DIR / "projects.json"
EVENTS_FILE = PRODUCT_DIR / "events.jsonl"
AUDITS_FILE = PRODUCT_DIR / "audits.json"
REVIEWS_FILE = PRODUCT_DIR / "reviews.json"
INTERVENTIONS_FILE = PRODUCT_DIR / "interventions.json"
REPORTS_DIR = PRODUCT_DIR / "reports"

REQUIRED_FIELDS = [
    "user_id",
    "task_id",
    "initial_answer",
    "initial_confidence",
    "ai_advice",
    "ai_confidence",
    "final_answer",
    "ground_truth",
    "context",
]

FIELD_ALIASES: dict[str, list[str]] = {
    "user_id": ["agent_id", "learner_id", "reviewer_id", "user", "participant_id", "student_id"],
    "task_id": ["ticket_id", "question_id", "case_id", "task", "item_id", "problem_id"],
    "initial_answer": ["human_first_decision", "initial_decision", "pre_answer", "first_answer"],
    "initial_confidence": ["confidence_before_ai", "pre_confidence", "human_confidence"],
    "ai_advice": ["model_recommendation", "ai_recommendation", "advice", "recommendation"],
    "ai_confidence": ["model_confidence", "ai_confidence_score", "recommendation_confidence"],
    "final_answer": ["human_final_decision", "post_answer", "final_decision"],
    "ground_truth": ["true_label", "qa_outcome", "test_result", "correct_answer", "label"],
    "context": ["domain", "task_type", "model_name", "context_json", "category"],
}

Action = Literal["allow", "request_verification", "show_uncertainty", "delay", "route_to_review"]


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    mode: Literal["audit", "shadow", "guardrail"] = "audit"
    owner: str = "local"


class Project(BaseModel):
    project_id: str
    name: str
    description: str = ""
    mode: Literal["audit", "shadow", "guardrail"] = "audit"
    created_at: str
    saved_thresholds: dict[str, float] = Field(default_factory=lambda: {"default_alpha": 0.10})
    saved_policies: list[str] = Field(default_factory=lambda: ["request_verification", "compare_evidence", "route_to_review"])
    model_versions: list[str] = Field(default_factory=lambda: ["ReliaGuard-NS local heuristic v0.1"])
    reviewer_notes: list[dict[str, Any]] = Field(default_factory=list)
    audit_reports: list[str] = Field(default_factory=list)
    export_history: list[dict[str, Any]] = Field(default_factory=list)


class GuardrailCheck(BaseModel):
    project_id: str = "default"
    user_id: str = "anonymous"
    task_id: str = "ad_hoc"
    initial_answer: str
    initial_confidence: float = Field(0.5, ge=0.0, le=1.0)
    ai_advice: str
    ai_confidence: float = Field(0.75, ge=0.0, le=1.0)
    final_answer: str | None = None
    ground_truth: str | None = None
    context: dict[str, Any] | str = Field(default_factory=dict)
    mode: Literal["audit", "shadow", "guardrail"] = "guardrail"


class GuardrailResponse(BaseModel):
    state: str
    risk: float
    uncertainty: float
    recommended_action: Action
    message: str
    active_rules: list[str]
    case_id: str
    mode: str
    intervention_template: str
    safety_boundary: str


class InteractionEvent(GuardrailCheck):
    final_answer: str
    ground_truth: str
    timestamp: str | None = None


class IngestRequest(BaseModel):
    project_id: str = "default"
    source_name: str = "uploaded_logs"
    records: list[dict[str, Any]]
    mapping: dict[str, str]
    file_type: Literal["csv", "jsonl", "json", "parquet", "manual"] = "manual"


class ReviewLabel(BaseModel):
    project_id: str = "default"
    case_id: str
    label: Literal[
        "true_harmful_reliance",
        "false_positive",
        "uncertain",
        "bad_ground_truth",
        "intervention_worked",
        "intervention_too_burdensome",
    ]
    reviewer: str = "local_reviewer"
    note: str = ""


class ReplayRequest(BaseModel):
    project_id: str = "default"
    alpha: float = Field(0.10, ge=0.01, le=0.50)
    policy: Literal["confidence_threshold", "symbolic_rule", "reliaguard_ns"] = "reliaguard_ns"


class InterventionTemplate(BaseModel):
    key: str
    risk_pattern: str
    intervention: str
    action: Action
    editable: bool = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_product_dir() -> None:
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "project"


def _read_json(path: Path, default: Any) -> Any:
    _ensure_product_dir()
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    _ensure_product_dir()
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    _ensure_product_dir()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _read_events(project_id: str | None = None) -> list[dict[str, Any]]:
    _ensure_product_dir()
    if not EVENTS_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    with EVENTS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if project_id is None or row.get("project_id") == project_id:
                rows.append(row)
    return rows


def default_projects() -> list[Project]:
    examples = [
        ("customer-support-copilot", "Customer Support Copilot", "Refund, escalation, and QA decision support."),
        ("ai-tutor-algebra-study", "AI Tutor Algebra Study", "Student solution, tutor hint, and final answer audits."),
        ("code-review-assistant", "Code Review Assistant", "Developer review decisions assisted by an AI reviewer."),
        ("loan-explanation-interface-demo", "Loan Explanation Interface Demo", "Synthetic demo workspace for explanation UI review."),
    ]
    return [
        Project(project_id=project_id, name=name, description=description, created_at=_now())
        for project_id, name, description in examples
    ]


def list_projects() -> list[dict[str, Any]]:
    projects = _read_json(PROJECTS_FILE, None)
    if projects is None:
        projects = [project.model_dump() for project in default_projects()]
        _write_json(PROJECTS_FILE, projects)
    return projects


def create_project(payload: ProjectCreate) -> dict[str, Any]:
    projects = list_projects()
    base = _slug(payload.name)
    project_id = base
    suffix = 2
    existing = {project["project_id"] for project in projects}
    while project_id in existing:
        project_id = f"{base}-{suffix}"
        suffix += 1
    project = Project(
        project_id=project_id,
        name=payload.name,
        description=payload.description,
        mode=payload.mode,
        created_at=_now(),
    )
    projects.append(project.model_dump())
    _write_json(PROJECTS_FILE, projects)
    return project.model_dump()


def get_project(project_id: str) -> dict[str, Any] | None:
    for project in list_projects():
        if project["project_id"] == project_id:
            return project
    return None


def _context_to_string(context: dict[str, Any] | str) -> str:
    if isinstance(context, str):
        return context
    parts = [str(value) for value in context.values() if value not in (None, "")]
    return " | ".join(parts) or "general decision support"


def _case_id(payload: dict[str, Any]) -> str:
    raw = "|".join(str(payload.get(key, "")) for key in ["project_id", "user_id", "task_id", "initial_answer", "ai_advice", "final_answer", "ground_truth"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _reliance_case_from_guardrail(payload: GuardrailCheck) -> RelianceCase:
    final_answer = payload.final_answer if payload.final_answer is not None else payload.ai_advice
    ground_truth = payload.ground_truth if payload.ground_truth is not None else payload.initial_answer
    return RelianceCase(
        initial_answer=payload.initial_answer,
        initial_confidence=payload.initial_confidence,
        ai_advice=payload.ai_advice,
        final_answer=final_answer,
        ground_truth=ground_truth,
        task_context=_context_to_string(payload.context),
        advice_source="AI",
        model_confidence=payload.ai_confidence,
    )


def intervention_templates() -> list[dict[str, Any]]:
    existing = _read_json(INTERVENTIONS_FILE, None)
    if existing is not None:
        return existing
    templates = [
        InterventionTemplate(
            key="high_overreliance",
            risk_pattern="High overreliance",
            intervention="Before accepting this AI suggestion, compare it against one piece of evidence.",
            action="request_verification",
        ),
        InterventionTemplate(
            key="high_underreliance",
            risk_pattern="High underreliance",
            intervention="The AI may be correct here. Review the supporting evidence before rejecting it.",
            action="show_uncertainty",
        ),
        InterventionTemplate(
            key="low_confidence_disagreement",
            risk_pattern="Low confidence plus disagreement",
            intervention="Slow down and inspect the disagreement before finalizing.",
            action="delay",
        ),
        InterventionTemplate(
            key="high_model_confidence_wrong_prone",
            risk_pattern="High model confidence on wrong-prone task",
            intervention="Request a second opinion or source check before finalizing.",
            action="route_to_review",
        ),
        InterventionTemplate(
            key="uncertain_state",
            risk_pattern="Uncertain state",
            intervention="Route this case to human review.",
            action="route_to_review",
        ),
    ]
    data = [template.model_dump() for template in templates]
    _write_json(INTERVENTIONS_FILE, data)
    return data


def _action_from_prediction(prediction: ReliancePrediction) -> Action:
    if prediction.state == "harmful_overreliance":
        return "request_verification"
    if prediction.state == "harmful_underreliance":
        return "show_uncertainty"
    if prediction.uncertainty >= 0.80:
        return "route_to_review"
    if prediction.uncertainty >= 0.68:
        return "delay"
    return "allow"


def _message_for_action(action: Action, prediction: ReliancePrediction) -> str:
    if action == "allow":
        return "Allow the decision to continue while logging the reliance trace."
    if action == "request_verification":
        return "Ask the user to compare evidence before accepting the AI recommendation."
    if action == "show_uncertainty":
        return "Show calibrated uncertainty and ask the user to inspect why the advice differs."
    if action == "delay":
        return "Delay finalization long enough for evidence comparison or a confidence check."
    return "Route this case to a reviewer because the reliance state is uncertain or high risk."


def _template_for_action(action: Action, prediction: ReliancePrediction) -> str:
    templates = intervention_templates()
    if prediction.state == "harmful_overreliance":
        key = "high_overreliance"
    elif prediction.state == "harmful_underreliance":
        key = "high_underreliance"
    elif action == "delay":
        key = "low_confidence_disagreement"
    elif action == "route_to_review":
        key = "uncertain_state"
    else:
        return "No intervention needed; continue logging in audit or shadow mode."
    for template in templates:
        if template["key"] == key:
            return template["intervention"]
    return _message_for_action(action, prediction)


def guardrail_check(payload: GuardrailCheck, persist: bool = True) -> dict[str, Any]:
    data = payload.model_dump()
    case_id = _case_id(data)
    if payload.ground_truth is None:
        disagreement = str(payload.initial_answer).strip().casefold() != str(payload.ai_advice).strip().casefold()
        active_rules = ["ground_truth_unavailable"]
        if disagreement:
            active_rules.append("human_advice_disagreement")
        if payload.ai_confidence >= 0.80:
            active_rules.append("high_model_confidence")
        if payload.initial_confidence >= 0.75:
            active_rules.append("high_initial_confidence")
        risk = max(0.05, min(0.95, 0.28 + (0.18 if disagreement else -0.05) + 0.20 * payload.ai_confidence + 0.12 * payload.initial_confidence))
        uncertainty = max(0.70, min(0.98, 1.0 - abs(payload.ai_confidence - payload.initial_confidence) + 0.10))
        action: Action = "route_to_review" if uncertainty >= 0.88 else "delay" if disagreement else "show_uncertainty"
        message = "Ground truth is not available yet; slow down, expose uncertainty, and preserve the case for later QA review."
        intervention_template = "Ask the user to compare evidence and log the case for post-hoc ground-truth audit."
        response = GuardrailResponse(
            state="uncertain_disagreement",
            risk=round(float(risk), 3),
            uncertainty=round(float(uncertainty), 3),
            recommended_action=action,
            message=message,
            active_rules=active_rules,
            case_id=case_id,
            mode=payload.mode,
            intervention_template=intervention_template,
            safety_boundary="Guardrail output is decision support and observability. Ground-truth-free checks are proxy risk estimates, not confirmed overreliance labels.",
        ).model_dump()
    else:
        case = _reliance_case_from_guardrail(payload)
        prediction = predict_reliance_case(case)
        action = _action_from_prediction(prediction)
        response = GuardrailResponse(
            state=prediction.state,
            risk=prediction.harmful_reliance_risk,
            uncertainty=prediction.uncertainty,
            recommended_action=action,
            message=_message_for_action(action, prediction),
            active_rules=prediction.active_rules,
            case_id=case_id,
            mode=payload.mode,
            intervention_template=_template_for_action(action, prediction),
            safety_boundary="Guardrail output is decision support and observability. It is not a causal or clinical claim.",
        ).model_dump()
    if persist:
        event = {**data, **response, "timestamp": _now(), "source": "guardrail_check"}
        _append_jsonl(EVENTS_FILE, event)
        if response["recommended_action"] != "allow" or response["risk"] >= 0.65:
            _upsert_review_case(event)
    return response


def log_interaction(payload: InteractionEvent) -> dict[str, Any]:
    check = GuardrailCheck(**payload.model_dump())
    response = guardrail_check(check, persist=False)
    event = {**payload.model_dump(), **response, "timestamp": payload.timestamp or _now(), "source": "sdk_event"}
    _append_jsonl(EVENTS_FILE, event)
    if response["recommended_action"] != "allow" or response["risk"] >= 0.65:
        _upsert_review_case(event)
    return {"stored": True, "case_id": response["case_id"], "guardrail": response}


def _infer_mapping(columns: list[str]) -> dict[str, str]:
    lower = {column.lower(): column for column in columns}
    mapping: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        if field in lower:
            mapping[field] = lower[field]
            continue
        for alias in FIELD_ALIASES[field]:
            if alias.lower() in lower:
                mapping[field] = lower[alias.lower()]
                break
    return mapping


def preview_columns(records: list[dict[str, Any]]) -> dict[str, Any]:
    columns = sorted({key for row in records for key in row.keys()})
    return {"columns": columns, "suggested_mapping": _infer_mapping(columns), "required_fields": REQUIRED_FIELDS, "aliases": FIELD_ALIASES}


def _to_float(value: Any, default: float = 0.5) -> float:
    try:
        if value in (None, ""):
            return default
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _canonicalize(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        source = mapping.get(field, "")
        canonical[field] = row.get(source) if source else None
    canonical["initial_confidence"] = _to_float(canonical.get("initial_confidence"), 0.5)
    canonical["ai_confidence"] = _to_float(canonical.get("ai_confidence"), 0.75)
    canonical["context"] = canonical.get("context") or "uploaded logs"
    return canonical


def _analyses_possible(mapping: dict[str, str]) -> list[str]:
    present = {field for field, source in mapping.items() if source}
    analyses = []
    if {"initial_answer", "ai_advice", "final_answer", "ground_truth"}.issubset(present):
        analyses.append("reliance_state_audit")
        analyses.append("overreliance_underreliance_rates")
        analyses.append("historical_policy_replay")
    if {"initial_confidence", "ai_confidence"}.issubset(present):
        analyses.append("confidence_and_calibration_summary")
    if "context" in present:
        analyses.append("risk_by_context_or_task_category")
    if "user_id" in present:
        analyses.append("user_or_cohort_heterogeneity")
    if "task_id" in present:
        analyses.append("case_review_queue")
    return analyses


def ingest_records(payload: IngestRequest) -> dict[str, Any]:
    preview = preview_columns(payload.records)
    missing = [field for field in REQUIRED_FIELDS if not payload.mapping.get(field)]
    canonical_rows = [_canonicalize(row, payload.mapping) for row in payload.records]
    scored: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    risks: list[float] = []
    for row in canonical_rows:
        if row.get("initial_answer") is None or row.get("ai_advice") is None or row.get("final_answer") is None or row.get("ground_truth") is None:
            continue
        event = InteractionEvent(
            project_id=payload.project_id,
            user_id=str(row.get("user_id") or "unknown"),
            task_id=str(row.get("task_id") or "unknown"),
            initial_answer=str(row["initial_answer"]),
            initial_confidence=float(row["initial_confidence"]),
            ai_advice=str(row["ai_advice"]),
            ai_confidence=float(row["ai_confidence"]),
            final_answer=str(row["final_answer"]),
            ground_truth=str(row["ground_truth"]),
            context={"source": payload.source_name, "context": row.get("context")},
        )
        logged = log_interaction(event)
        guardrail = logged["guardrail"]
        scored_row = {**row, **guardrail, "project_id": payload.project_id, "source_name": payload.source_name}
        scored.append(scored_row)
        state_counts[guardrail["state"]] = state_counts.get(guardrail["state"], 0) + 1
        action_counts[guardrail["recommended_action"]] = action_counts.get(guardrail["recommended_action"], 0) + 1
        risks.append(float(guardrail["risk"]))
    harmful = [row for row in scored if row["state"] in {"harmful_overreliance", "harmful_underreliance"}]
    audit_id = f"audit-{hashlib.sha1((payload.project_id + payload.source_name + _now()).encode('utf-8')).hexdigest()[:10]}"
    report = {
        "audit_id": audit_id,
        "project_id": payload.project_id,
        "source_name": payload.source_name,
        "file_type": payload.file_type,
        "uploaded_records": len(payload.records),
        "scored_records": len(scored),
        "missing_fields": missing,
        "analyses_possible": _analyses_possible(payload.mapping),
        "state_distribution": state_counts,
        "action_distribution": action_counts,
        "overreliance_rate": state_counts.get("harmful_overreliance", 0) / max(1, len(scored)),
        "underreliance_rate": state_counts.get("harmful_underreliance", 0) / max(1, len(scored)),
        "mean_risk": sum(risks) / max(1, len(risks)),
        "highest_risk_cases": sorted(scored, key=lambda row: row["risk"], reverse=True)[:5],
        "threshold_recommendation": "Start in shadow mode at alpha=0.10, review false positives, then lower alpha only if burden is acceptable.",
        "limitations": [
            "Ground truth quality determines label quality.",
            "Uploaded logs support observational audit and replay, not causal proof of intervention effectiveness.",
            "Missing initial/final answer fields limit reliance-state analysis.",
        ],
        "created_at": _now(),
        "column_preview": preview,
    }
    audits = _read_json(AUDITS_FILE, [])
    audits.append(report)
    _write_json(AUDITS_FILE, audits)
    _write_report_html(report)
    return report


async def parse_upload(file: UploadFile) -> dict[str, Any]:
    filename = file.filename or "uploaded"
    suffix = Path(filename).suffix.lower()
    content = await file.read()
    if suffix == ".csv":
        text = content.decode("utf-8-sig")
        records = list(csv.DictReader(io.StringIO(text)))
        return {"file_type": "csv", "records": records, **preview_columns(records)}
    if suffix in {".jsonl", ".ndjson"}:
        text = content.decode("utf-8-sig")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {"file_type": "jsonl", "records": records, **preview_columns(records)}
    if suffix == ".json":
        parsed = json.loads(content.decode("utf-8-sig"))
        records = parsed if isinstance(parsed, list) else parsed.get("records", [])
        return {"file_type": "json", "records": records, **preview_columns(records)}
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("Parquet preview requires pandas with parquet support.") from exc
        frame = pd.read_parquet(io.BytesIO(content))
        records = frame.head(5000).to_dict(orient="records")
        return {"file_type": "parquet", "records": records, **preview_columns(records)}
    raise ValueError("Unsupported file type. Upload CSV, JSON, JSONL, or Parquet.")


def _upsert_review_case(event: dict[str, Any]) -> None:
    reviews = _read_json(REVIEWS_FILE, [])
    case_id = event["case_id"]
    if any(item.get("case_id") == case_id for item in reviews):
        return
    reviews.append(
        {
            "case_id": case_id,
            "project_id": event.get("project_id", "default"),
            "task_id": event.get("task_id", "unknown"),
            "user_id": event.get("user_id", "unknown"),
            "state": event.get("state"),
            "risk": event.get("risk"),
            "action": event.get("recommended_action"),
            "reviewer_decision": "needs_review",
            "review_history": [],
            "created_at": _now(),
            "excerpt": {
                "initial_answer": event.get("initial_answer"),
                "ai_advice": event.get("ai_advice"),
                "final_answer": event.get("final_answer"),
                "ground_truth": event.get("ground_truth"),
            },
        }
    )
    _write_json(REVIEWS_FILE, reviews)


def review_queue(project_id: str | None = None) -> list[dict[str, Any]]:
    reviews = _read_json(REVIEWS_FILE, [])
    if project_id:
        reviews = [row for row in reviews if row.get("project_id") == project_id]
    return sorted(reviews, key=lambda row: float(row.get("risk") or 0), reverse=True)


def label_review(payload: ReviewLabel) -> dict[str, Any]:
    reviews = _read_json(REVIEWS_FILE, [])
    for row in reviews:
        if row.get("case_id") == payload.case_id and row.get("project_id") == payload.project_id:
            row["reviewer_decision"] = payload.label
            row.setdefault("review_history", []).append({**payload.model_dump(), "timestamp": _now()})
            _write_json(REVIEWS_FILE, reviews)
            return row
    raise KeyError(f"Case {payload.case_id} not found")


def _write_report_html(report: dict[str, Any]) -> Path:
    path = REPORTS_DIR / f"{report['audit_id']}.html"
    highest = "".join(
        f"<li><strong>{case.get('task_id', 'case')}</strong>: {case.get('state')} risk={case.get('risk'):.2f}, action={case.get('recommended_action')}</li>"
        for case in report.get("highest_risk_cases", [])
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ReliaGuard Audit Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:40px;max-width:980px;color:#111a1f}} .card{{border:1px solid #ddd;border-radius:18px;padding:18px;margin:14px 0}} code{{background:#f4ead9;padding:2px 6px;border-radius:6px}}</style></head>
<body>
<h1>ReliaGuard Audit Report</h1>
<p><strong>Project:</strong> {report['project_id']} | <strong>Source:</strong> {report['source_name']} | <strong>Created:</strong> {report['created_at']}</p>
<div class="card"><h2>Dataset summary</h2><p>{report['scored_records']} scored records from {report['uploaded_records']} uploaded rows.</p></div>
<div class="card"><h2>Reliance-state distribution</h2><pre>{json.dumps(report['state_distribution'], indent=2)}</pre></div>
<div class="card"><h2>Reliance failures</h2><p>Overreliance rate: {report['overreliance_rate']:.1%}. Underreliance rate: {report['underreliance_rate']:.1%}. Mean harmful-risk score: {report['mean_risk']:.2f}.</p></div>
<div class="card"><h2>Highest-risk cases</h2><ul>{highest}</ul></div>
<div class="card"><h2>Threshold recommendation</h2><p>{report['threshold_recommendation']}</p></div>
<div class="card"><h2>Limitations</h2><ul>{"".join(f"<li>{item}</li>" for item in report['limitations'])}</ul></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


def reports(project_id: str | None = None) -> list[dict[str, Any]]:
    rows = _read_json(AUDITS_FILE, [])
    if project_id:
        rows = [row for row in rows if row.get("project_id") == project_id]
    return rows


def replay_logs(payload: ReplayRequest) -> dict[str, Any]:
    events = _read_events(payload.project_id)
    if payload.policy == "confidence_threshold":
        threshold = 0.82
    elif payload.policy == "symbolic_rule":
        threshold = 0.65
    else:
        threshold = max(0.20, min(0.85, 0.25 + 0.90 * payload.alpha))
    scored = [event for event in events if "risk" in event]
    flagged = [event for event in scored if float(event.get("risk") or 0) >= threshold]
    harmful = [event for event in scored if event.get("state") in {"harmful_overreliance", "harmful_underreliance"}]
    caught = [event for event in flagged if event.get("state") in {"harmful_overreliance", "harmful_underreliance"}]
    normal_interrupted = [event for event in flagged if event not in caught]
    return {
        "project_id": payload.project_id,
        "alpha": payload.alpha,
        "policy": payload.policy,
        "threshold": threshold,
        "total_cases": len(scored),
        "flagged_cases": len(flagged),
        "harmful_cases": len(harmful),
        "harmful_cases_caught": len(caught),
        "harmful_capture": len(caught) / max(1, len(harmful)),
        "missed_harmful": len(harmful) - len(caught),
        "normal_cases_interrupted": len(normal_interrupted),
        "intervention_burden": len(flagged) / max(1, len(scored)),
        "examples": flagged[:6],
        "boundary": "Historical replay is observational. It estimates what the gate would have flagged, not how people would have changed behavior.",
    }


def monitoring_summary(project_id: str | None = None) -> dict[str, Any]:
    events = _read_events(project_id)
    if not events:
        return {"events": 0, "timeline": [], "by_context": [], "by_model": [], "review_feedback": {}, "message": "No streamed or uploaded events yet."}
    state_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    context_counts: dict[str, dict[str, float]] = {}
    for event in events:
        state = str(event.get("state", "unknown"))
        action = str(event.get("recommended_action", "unknown"))
        state_counts[state] = state_counts.get(state, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        context = event.get("context")
        if isinstance(context, dict):
            key = str(context.get("domain") or context.get("context") or context.get("source") or "unknown")
        else:
            key = str(context or "unknown")
        bucket = context_counts.setdefault(key, {"cases": 0, "risk_sum": 0.0, "over": 0, "under": 0})
        bucket["cases"] += 1
        bucket["risk_sum"] += float(event.get("risk") or 0)
        bucket["over"] += 1 if state == "harmful_overreliance" else 0
        bucket["under"] += 1 if state == "harmful_underreliance" else 0
    reviews = review_queue(project_id)
    review_feedback: dict[str, int] = {}
    for row in reviews:
        label = row.get("reviewer_decision", "needs_review")
        review_feedback[label] = review_feedback.get(label, 0) + 1
    by_context = [
        {
            "context": key,
            "cases": int(value["cases"]),
            "mean_risk": value["risk_sum"] / max(1, value["cases"]),
            "overreliance_rate": value["over"] / max(1, value["cases"]),
            "underreliance_rate": value["under"] / max(1, value["cases"]),
        }
        for key, value in context_counts.items()
    ]
    return {
        "events": len(events),
        "state_counts": state_counts,
        "action_counts": action_counts,
        "by_context": sorted(by_context, key=lambda row: row["mean_risk"], reverse=True),
        "review_feedback": review_feedback,
    }


def demo_customer_support_records() -> list[dict[str, Any]]:
    return [
        {
            "agent_id": "agent_123",
            "ticket_id": "ticket_381",
            "human_first_decision": "refund",
            "confidence_before_ai": 0.72,
            "model_recommendation": "deny_refund",
            "model_confidence": 0.88,
            "human_final_decision": "deny_refund",
            "true_label": "refund",
            "domain": "customer_support",
        },
        {
            "agent_id": "agent_177",
            "ticket_id": "ticket_418",
            "human_first_decision": "ask_for_more_information",
            "confidence_before_ai": 0.44,
            "model_recommendation": "escalate",
            "model_confidence": 0.74,
            "human_final_decision": "escalate",
            "true_label": "escalate",
            "domain": "customer_support",
        },
        {
            "agent_id": "agent_212",
            "ticket_id": "ticket_456",
            "human_first_decision": "deny_refund",
            "confidence_before_ai": 0.66,
            "model_recommendation": "refund",
            "model_confidence": 0.82,
            "human_final_decision": "deny_refund",
            "true_label": "refund",
            "domain": "customer_support",
        },
    ]
