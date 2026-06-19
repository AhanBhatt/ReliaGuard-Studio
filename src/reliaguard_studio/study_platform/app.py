from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .advice_generator import generate_advice
from .randomization import assign_condition
from .schema import Participant, ParticipantCreate, ResponseRecord
from .storage import connect
from .tasks import TASK_BANK, get_task

app = FastAPI(title="ReliaGuard-NS Prospective Validation Platform")


@app.post("/participants", response_model=Participant)
def create_participant(payload: ParticipantCreate) -> Participant:
    if not payload.consent_confirmed or not payload.age_over_18:
        raise HTTPException(status_code=400, detail="Consent and age confirmation are required.")
    participant = Participant(condition=assign_condition(payload.self_reported_ai_use or "participant"))
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO participants(participant_id, condition, created_at) VALUES (?, ?, ?)",
            (participant.participant_id, participant.condition, participant.created_at.isoformat()),
        )
        conn.commit()
    return participant


@app.get("/tasks")
def list_tasks() -> list[dict[str, str]]:
    return [task.model_dump() for task in TASK_BANK]


@app.get("/tasks/{task_id}")
def read_task(task_id: str) -> dict[str, str]:
    try:
        return get_task(task_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc


@app.get("/participants/{participant_id}/tasks/{task_id}/advice")
def read_advice(participant_id: str, task_id: str, condition: str) -> dict[str, str | bool]:
    try:
        task = get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    if condition not in {"no_gating", "confidence_threshold", "symbolic_rule", "reliaguard_ns"}:
        raise HTTPException(status_code=400, detail="Unknown study condition")
    advice = generate_advice(task, participant_id, condition)  # type: ignore[arg-type]
    return {
        "task_id": advice.task_id,
        "advice_label": advice.advice_label,
        "advice_correct": advice.advice_correct,
        "advice_source": advice.advice_source,
        "disclosure": advice.disclosure,
    }


@app.post("/responses")
def write_response(record: ResponseRecord) -> dict[str, str]:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO responses(
                participant_id, task_id, condition, initial_answer, initial_confidence,
                advice_label, final_answer, final_confidence, verification_action, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.participant_id,
                record.task_id,
                record.condition,
                record.initial_answer,
                record.initial_confidence,
                record.advice_label,
                record.final_answer,
                record.final_confidence,
                record.verification_action,
                record.timestamp.isoformat(),
            ),
        )
        conn.commit()
    return {"status": "stored"}
