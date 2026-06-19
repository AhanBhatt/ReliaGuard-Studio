from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..config.schemas import ProjectConfig, TaskFamilyConfig


CONCEPT_BANK = {
    "reading_comprehension": [
        ("ecosystem recovery", "A wetland restoration project improved water quality after invasive plants were removed."),
        ("battery chemistry", "Solid-state batteries reduce leakage risk but currently cost more to manufacture."),
        ("public health", "Vaccination campaigns reduce severe disease, but outreach quality shapes uptake."),
        ("astronomy", "Exoplanet transits help estimate planetary size when brightness dips are measured carefully."),
    ],
    "scientific_reasoning": [
        ("enzyme kinetics", "A mutant enzyme doubles activity at low temperature but destabilizes above 35C."),
        ("agriculture trial", "A fertilizer increases yield in clay soil but not in sandy soil."),
        ("psychology study", "Participants who self-explain examples transfer concepts better than those who reread."),
        ("air-quality study", "Sensors near traffic corridors overestimate citywide exposure if rural controls are absent."),
    ],
    "coding": [
        ("python indexing", "A loop stops one element early because the range upper bound is exclusive."),
        ("sql filtering", "A query duplicates rows after an unintended many-to-many join."),
        ("api retries", "A client retries non-idempotent POST requests and causes duplicate writes."),
        ("data leakage", "A preprocessing step fits on the full dataset before the train/test split."),
    ],
    "mathematical_reasoning": [
        ("fractions", "A recipe scaled by three quarters changes the ratio of sugar to flour."),
        ("probability", "A diagnostic test has strong sensitivity but low base-rate prevalence."),
        ("geometry", "A rectangle perimeter is fixed while area changes under different side lengths."),
        ("combinatorics", "A committee is formed with one representative from each subgroup."),
    ],
    "writing_synthesis": [
        ("climate adaptation", "Two reports disagree on whether seawalls or retreat is more sustainable."),
        ("ai policy", "One source prioritizes innovation speed while another prioritizes accountability."),
        ("education technology", "A school report shows productivity gains but mixed long-term learning outcomes."),
        ("scientific communication", "Researchers balance accuracy, clarity, and public trust."),
    ],
    "source_verification": [
        ("citation mismatch", "A quoted statistic differs subtly from the cited source wording."),
        ("secondary source", "A news article paraphrases a study but omits a limitation."),
        ("outdated evidence", "A cited benchmark result predates a major dataset revision."),
        ("confounded claim", "A source states correlation while the answer implies causation."),
    ],
    "false_answer_detection": [
        ("unit conversion", "A model gives a numerically plausible answer but mixes meters and centimeters."),
        ("code explanation", "A model claims code is O(n) when nested iteration makes it O(n^2)."),
        ("study interpretation", "A model overstates a result by treating an observational study as causal."),
        ("source summary", "A model attributes a claim to the wrong source in a synthesis task."),
    ],
}


@dataclass
class TaskRecord:
    task_id: str
    family: str
    concept: str
    prompt: str
    reference_answer: str
    source_stub: str
    flawed_ai_answer: str
    transfer_prompt: str
    recall_prompt: str
    difficulty: float


def _build_task_text(task_family: TaskFamilyConfig, concept: str, scenario: str, difficulty: float, index: int) -> TaskRecord:
    family_id = task_family.id
    base_prompt = f"[{task_family.name}] Scenario {index}: {scenario} Explain the key idea and justify your answer."
    reference = (
        f"The correct answer should address the {concept} concept, reason about the scenario directly, and avoid copying unsupported claims."
    )
    source_stub = f"Source note for {concept}: this synthetic reference includes one supporting detail and one limitation."
    flawed_answer = (
        f"A plausible but flawed AI response about {concept} that sounds confident, omits the key limitation, and subtly misinterprets the scenario."
    )
    transfer_prompt = (
        f"Apply the core idea from {concept} to a new but related problem without AI assistance. Difficulty={difficulty:.2f}."
    )
    recall_prompt = f"Without looking back, restate the main principle behind {concept} in your own words."
    return TaskRecord(
        task_id=f"{family_id}_{index:03d}",
        family=family_id,
        concept=concept,
        prompt=base_prompt,
        reference_answer=reference,
        source_stub=source_stub,
        flawed_ai_answer=flawed_answer,
        transfer_prompt=transfer_prompt,
        recall_prompt=recall_prompt,
        difficulty=float(difficulty),
    )


def build_air_bench_catalog(config: ProjectConfig, seed: int | None = None, tasks_per_family: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed or config.simulation.seed)
    records: list[TaskRecord] = []
    for family in config.task_families:
        concepts: Iterable[tuple[str, str]] = CONCEPT_BANK[family.id]
        for idx in range(tasks_per_family):
            concept, scenario = list(concepts)[idx % len(CONCEPT_BANK[family.id])]
            difficulty = rng.uniform(*family.difficulty_range)
            records.append(_build_task_text(family, concept, scenario, difficulty, idx + 1))
    return pd.DataFrame([record.__dict__ for record in records])
