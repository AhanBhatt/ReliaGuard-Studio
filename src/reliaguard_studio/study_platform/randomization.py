from __future__ import annotations

from hashlib import sha256

from .schema import StudyCondition


CONDITIONS: tuple[StudyCondition, ...] = ("no_gating", "confidence_threshold", "symbolic_rule", "reliaguard_ns")


def assign_condition(participant_seed: str) -> StudyCondition:
    digest = sha256(participant_seed.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(CONDITIONS)
    return CONDITIONS[idx]

