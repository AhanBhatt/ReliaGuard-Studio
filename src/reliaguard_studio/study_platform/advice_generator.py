from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .schema import StudyCondition, TrialTask


@dataclass(frozen=True)
class AdviceRecord:
    task_id: str
    advice_label: str
    advice_correct: bool
    advice_source: str
    disclosure: str


def generate_advice(task: TrialTask, participant_id: str, condition: StudyCondition) -> AdviceRecord:
    """Deterministically generate balanced correct/flawed advice for study dry-runs.

    The prospective platform uses this helper to lock the advice schedule before
    deployment. It does not call external AI services and it does not generate
    participant evidence.
    """

    digest = sha256(f"{participant_id}:{task.task_id}:{condition}".encode("utf-8")).hexdigest()
    advice_correct = int(digest[:8], 16) % 2 == 0
    label = task.correct_advice_label if advice_correct else task.flawed_advice_label
    return AdviceRecord(
        task_id=task.task_id,
        advice_label=label,
        advice_correct=advice_correct,
        advice_source="prelocked study advice",
        disclosure="Advice may be correct or incorrect. Participants should verify before finalizing.",
    )
