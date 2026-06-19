from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


StudyCondition = Literal["no_gating", "confidence_threshold", "symbolic_rule", "reliaguard_ns"]


class ParticipantCreate(BaseModel):
    consent_confirmed: bool
    age_over_18: bool
    self_reported_ai_use: str | None = None


class Participant(BaseModel):
    participant_id: str = Field(default_factory=lambda: uuid4().hex)
    condition: StudyCondition
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrialTask(BaseModel):
    task_id: str
    task_family: str
    prompt: str
    correct_label: str
    flawed_advice_label: str
    correct_advice_label: str


class ResponseRecord(BaseModel):
    participant_id: str
    task_id: str
    condition: StudyCondition
    initial_answer: str
    initial_confidence: float = Field(ge=0.0, le=1.0)
    advice_label: str
    final_answer: str
    final_confidence: float = Field(ge=0.0, le=1.0)
    verification_action: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

