from __future__ import annotations

from .schema import TrialTask


TASK_BANK = [
    TrialTask(
        task_id="source_verification_001",
        task_family="source_verification",
        prompt="A news summary claims that a medical guideline changed last month. Decide whether the claim is supported by the provided source excerpt.",
        correct_label="unsupported",
        flawed_advice_label="supported",
        correct_advice_label="unsupported",
    ),
    TrialTask(
        task_id="math_reasoning_001",
        task_family="mathematical_reasoning",
        prompt="A worked solution claims that 3(x-2)=18 implies x=4. Decide whether the solution is correct.",
        correct_label="incorrect",
        flawed_advice_label="correct",
        correct_advice_label="incorrect",
    ),
    TrialTask(
        task_id="coding_review_001",
        task_family="coding",
        prompt="A code assistant says a loop is safe because the list length never changes. Decide whether the advice should be accepted.",
        correct_label="needs_verification",
        flawed_advice_label="accept",
        correct_advice_label="needs_verification",
    ),
    TrialTask(
        task_id="microscopy_qc_001",
        task_family="life_science_quality_control",
        prompt="A microscopy-analysis assistant says a cell-count image is usable even though the excerpt reports saturated nuclei and uneven illumination. Decide whether the image should pass quality control.",
        correct_label="fail_qc",
        flawed_advice_label="pass_qc",
        correct_advice_label="fail_qc",
    ),
    TrialTask(
        task_id="genomics_interpretation_001",
        task_family="life_science_reasoning",
        prompt="A genomics assistant says a variant is causal because it is rare, but the excerpt states that segregation evidence and functional validation are absent. Decide whether the claim is supported.",
        correct_label="unsupported",
        flawed_advice_label="supported",
        correct_advice_label="unsupported",
    ),
    TrialTask(
        task_id="protocol_check_001",
        task_family="research_protocol_verification",
        prompt="A lab-protocol assistant recommends skipping a negative control because the positive control passed. Decide whether the protocol recommendation is acceptable.",
        correct_label="needs_control",
        flawed_advice_label="acceptable",
        correct_advice_label="needs_control",
    ),
]


def get_task(task_id: str) -> TrialTask:
    for task in TASK_BANK:
        if task.task_id == task_id:
            return task
    raise KeyError(task_id)
