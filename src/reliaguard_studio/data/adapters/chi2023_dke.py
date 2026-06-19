from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...paths import REAL_DATA_PREPARED_DIR
from ..real_schema import DatasetCard, PreparedRealDataset
from .registry import BaseRealDatasetAdapter, download_file, extract_zip


DOWNLOAD_URL = "https://data.4tu.nl/ndownloader/items/96010177-46e8-4967-9a49-fe38f0bace4e/versions/1"

QUESTION_ORDERS = {
    0: [1, 4, 5, 2, 3, 0, 7, 10, 8, 9, 14, 17, 13, 12, 16, 15],
    1: [12, 17, 13, 14, 16, 15, 9, 10, 8, 7, 5, 0, 2, 1, 4, 3],
    2: [1, 4, 5, 0, 3, 2, 8, 10, 9, 7, 15, 13, 17, 14, 16, 12],
    3: [12, 16, 17, 14, 13, 15, 10, 8, 9, 7, 2, 4, 5, 0, 3, 1],
    4: [3, 1, 4, 2, 0, 5, 8, 10, 9, 7, 17, 13, 15, 12, 16, 14],
    5: [16, 13, 14, 17, 12, 15, 9, 10, 8, 7, 0, 5, 1, 3, 4, 2],
    6: [4, 1, 3, 0, 2, 5, 8, 7, 10, 9, 17, 16, 13, 15, 14, 12],
    7: [12, 14, 16, 17, 15, 13, 8, 7, 10, 9, 3, 2, 4, 1, 5, 0],
    8: [2, 0, 4, 5, 3, 1, 8, 9, 7, 10, 16, 15, 17, 12, 13, 14],
    9: [12, 14, 16, 15, 13, 17, 10, 9, 8, 7, 0, 4, 5, 3, 2, 1],
}
QUESTION_COLUMNS = ["question0", "question1", "question2", "question3", "question4", "question5", "question7", "question8", "question9", "question10", "question12", "question13", "question14", "question15", "question16", "question17"]
ADVICE_COLUMNS = ["advice0", "advice1", "advice2", "advice3", "advice4", "advice5", "advice7", "advice8", "advice9", "advice10", "advice12", "advice13", "advice14", "advice15", "advice16", "advice17"]
ATTENTION_ANSWERS = {"attention_ati": 3, "attention6": "B", "attention11": "D", "attention18": "C"}


def _reverse_code(value: float, max_scale: float) -> float:
    return max_scale + 1 - value


def _questionnaire_mean(values: list[float], reverse_indices: list[int], max_scale: int, value_add_one: bool = True) -> float:
    converted = []
    for index, value in enumerate(values, start=1):
        value_ = float(value) + 1 if value_add_one else float(value)
        converted.append(_reverse_code(value_, max_scale) if index in reverse_indices else value_)
    return float(np.mean(converted))


class CHI2023DKEAdapter(BaseRealDatasetAdapter):
    dataset_card = DatasetCard(
        name="CHI 2023 appropriate reliance dataset",
        short_name="chi2023_dke",
        source="4TU ResearchData / GitHub",
        source_url="https://doi.org/10.4121/96010177-46E8-4967-9A49-FE38F0BACE4E",
        license_name="CC BY 4.0 in 4TU metadata; GitHub README contains older restrictive wording",
        citation=(
            "He G., Kuiper L., Gadiraju U. Knowing About Knowing: An Illusion of Human Competence "
            "Can Hinder Appropriate Reliance on AI Systems. CHI 2023."
        ),
        decision="integrated",
        auto_download=True,
        redistributable=False,
        participant_or_interaction_count="249 participants",
        domains=["logical reasoning"],
        supported_constructs=[
            "appropriate reliance",
            "overreliance",
            "underreliance",
            "self-assessment calibration",
            "tutorial intervention",
            "XAI condition effects",
        ],
        note=(
            "The DOI landing page exposes CC BY 4.0 metadata. The GitHub README contains older, more restrictive wording, "
            "so raw-data redistribution should be re-checked before release."
        ),
    )

    def download(self, force: bool = False) -> list[Path]:
        archive_path = download_file(DOWNLOAD_URL, self.raw_dir / "chi2023_dke.zip", force=force)
        extracted = extract_zip(archive_path, self.raw_dir)
        return [archive_path, *extracted]

    def prepare(self) -> PreparedRealDataset:
        self.download(force=False)
        response_path = self.raw_dir / "all_valid_data.csv"
        tasks_path = self.raw_dir / "selected_samples.csv"
        if not response_path.exists():
            response_path = self.raw_dir / "anonymous_data" / "all_valid_data.csv"
            tasks_path = self.raw_dir / "anonymous_data" / "selected_samples.csv"
        responses = pd.read_csv(response_path)
        tasks = pd.read_csv(tasks_path)
        responses = responses.dropna(axis=0).copy()

        attention_passes = (
            (responses["attention_ati"] == ATTENTION_ANSWERS["attention_ati"]).astype(int)
            + (responses["attention6"] == ATTENTION_ANSWERS["attention6"]).astype(int)
            + (responses["attention11"] == ATTENTION_ANSWERS["attention11"]).astype(int)
            + (responses["attention18"] == ATTENTION_ANSWERS["attention18"]).astype(int)
        )
        responses["attention_passes"] = attention_passes
        responses = responses.loc[responses["attention_passes"] >= 4].reset_index(drop=True)

        responses["dataset_name"] = "chi2023_dke"
        responses["participant_id"] = responses["username"].astype(str)
        responses["tutorial_present"] = responses["tutorial"].astype(int)
        responses["xai_present"] = responses["XAI"].astype(int)
        responses["condition_id"] = responses.apply(
            lambda row: f"tutorial_{int(row['tutorial'])}_xai_{int(row['XAI'])}", axis=1
        )
        responses["condition_name"] = responses.apply(
            lambda row: (
                "tutorial + xai"
                if row["tutorial"] == 1 and row["XAI"] == 1
                else "tutorial only"
                if row["tutorial"] == 1 and row["XAI"] == 0
                else "xai only"
                if row["tutorial"] == 0 and row["XAI"] == 1
                else "control"
            ),
            axis=1,
        )

        responses["ati_scale"] = responses.apply(
            lambda row: _questionnaire_mean(
                [row[f"ati{i}"] for i in range(1, 10)],
                reverse_indices=[3, 6, 8],
                max_scale=6,
            ),
            axis=1,
        )
        responses["propensity_to_trust"] = responses.apply(
            lambda row: _questionnaire_mean([row["pt1"], row["pt2"], row["pt3"]], reverse_indices=[1], max_scale=5),
            axis=1,
        )
        responses["trust_first"] = responses.apply(
            lambda row: _questionnaire_mean([row["tia1_1"], row["tia1_2"]], reverse_indices=[], max_scale=5),
            axis=1,
        )
        responses["trust_second"] = responses.apply(
            lambda row: _questionnaire_mean([row["tia2_1"], row["tia2_2"]], reverse_indices=[], max_scale=5),
            axis=1,
        )
        responses["xai_helpfulness_normalized"] = responses["xai_question"].replace(-1, np.nan).add(1).div(5.0)

        task_lookup = tasks[["answer", "id_string", "AI-advice"]].copy()
        task_lookup.index = tasks.index
        answer_key = {
            int(index): {
                "correct_label": row["answer"],
                "advice_label": row["AI-advice"],
                "task_id_string": row["id_string"],
            }
            for index, row in task_lookup.iterrows()
        }

        rows: list[dict[str, object]] = []
        participant_rows: list[dict[str, object]] = []
        for _, participant in responses.iterrows():
            participant_id = str(participant["participant_id"])
            question_order = QUESTION_ORDERS[int(participant["question_order"])]
            first_batch = set(question_order[:6])
            second_batch = set(question_order[-6:])

            actual_first_correct = 0
            actual_second_correct = 0
            preview_rows = []
            for q_col, a_col in zip(QUESTION_COLUMNS, ADVICE_COLUMNS, strict=True):
                task_id = int(q_col.replace("question", ""))
                task_meta = answer_key[task_id]
                initial_response = participant[q_col]
                final_response = participant[a_col]
                analysis_batch = "middle"
                analysis_included = False
                batch_index = np.nan
                if task_id in first_batch:
                    analysis_batch = "first"
                    analysis_included = True
                    batch_index = list(question_order[:6]).index(task_id)
                elif task_id in second_batch:
                    analysis_batch = "second"
                    analysis_included = True
                    batch_index = list(question_order[-6:]).index(task_id)

                initial_correct = int(initial_response == task_meta["correct_label"])
                final_correct = int(final_response == task_meta["correct_label"])
                advice_correct = int(task_meta["advice_label"] == task_meta["correct_label"])
                if analysis_batch == "first":
                    actual_first_correct += final_correct
                elif analysis_batch == "second":
                    actual_second_correct += final_correct

                preview_rows.append(
                    {
                        "dataset_name": "chi2023_dke",
                        "participant_id": participant_id,
                        "task_family": "logical_reasoning",
                        "domain_group": "logical_reasoning",
                        "task_instance_id": task_id,
                        "task_instance_key": task_meta["task_id_string"],
                        "condition_id": participant["condition_id"],
                        "condition_name": participant["condition_name"],
                        "tutorial_present": int(participant["tutorial_present"]),
                        "xai_present": int(participant["xai_present"]),
                        "advice_source_label": "ai",
                        "advice_source_ai": 1,
                        "initial_label": initial_response,
                        "final_label": final_response,
                        "advice_label": task_meta["advice_label"],
                        "correct_label": task_meta["correct_label"],
                        "initial_correct": initial_correct,
                        "initial_incorrect": 1 - initial_correct,
                        "final_correct": final_correct,
                        "advice_correct": advice_correct,
                        "advice_wrong": 1 - advice_correct,
                        "disagreement_case": int(initial_response != task_meta["advice_label"]),
                        "switch_behavior": int(initial_response != final_response),
                        "advice_uptake": int(initial_response != task_meta["advice_label"] and final_response == task_meta["advice_label"]),
                        "movement_toward_advice": int(initial_response != task_meta["advice_label"] and final_response == task_meta["advice_label"]),
                        "movement_away_from_advice": int(initial_response != task_meta["advice_label"] and final_response != task_meta["advice_label"]),
                        "human_team_improvement": final_correct - initial_correct,
                        "correct_ai_reliance": int(initial_correct == 0 and advice_correct == 1 and final_correct == 1),
                        "correct_self_reliance": int(initial_correct == 1 and advice_correct == 0 and final_correct == 1),
                        "overreliance": int(initial_correct == 1 and advice_correct == 0 and final_response == task_meta["advice_label"]),
                        "underreliance": int(initial_correct == 0 and advice_correct == 1 and final_response != task_meta["advice_label"]),
                        "beneficial_ai_reliance": int(initial_correct == 0 and advice_correct == 1 and final_response == task_meta["advice_label"]),
                        "harmful_reliance": 0,
                        "appropriate_reliance": np.nan if initial_response == task_meta["advice_label"] else final_correct,
                        "analysis_batch": analysis_batch,
                        "analysis_included": int(analysis_included),
                        "batch_position": batch_index,
                        "task_position_normalized": (float(batch_index) / 5.0) if analysis_included and batch_index == batch_index else np.nan,
                        "ati_scale": participant["ati_scale"],
                        "propensity_to_trust": participant["propensity_to_trust"],
                        "trust_first": participant["trust_first"],
                        "trust_second": participant["trust_second"],
                        "xai_helpfulness_normalized": participant["xai_helpfulness_normalized"],
                    }
                )

            first_gap = float(participant["surveySelf1"] - actual_first_correct)
            second_gap = float(participant["surveySelf2"] - actual_second_correct)
            first_group = "overestimation" if first_gap > 0 else "underestimation" if first_gap < 0 else "accurate"
            second_group = "overestimation" if second_gap > 0 else "underestimation" if second_gap < 0 else "accurate"

            for row in preview_rows:
                batch = row["analysis_batch"]
                if batch == "first":
                    row["self_assessment_count"] = float(participant["surveySelf1"])
                    row["peer_assessment_count"] = float(participant["surveyOther1"])
                    row["self_percentile"] = float(participant["surveyPercentage1"]) / 100.0
                    row["self_assessment_gap"] = first_gap
                    row["miscalibration_group"] = first_group
                    row["batch_trust"] = participant["trust_first"]
                elif batch == "second":
                    row["self_assessment_count"] = float(participant["surveySelf2"])
                    row["peer_assessment_count"] = float(participant["surveyOther2"])
                    row["self_percentile"] = float(participant["surveyPercentage2"]) / 100.0
                    row["self_assessment_gap"] = second_gap
                    row["miscalibration_group"] = second_group
                    row["batch_trust"] = participant["trust_second"]
                    row["first_batch_self_assessment_gap"] = first_gap
                    row["first_batch_overestimation"] = int(first_gap > 0)
                    row["first_batch_underestimation"] = int(first_gap < 0)
                else:
                    row["self_assessment_count"] = np.nan
                    row["peer_assessment_count"] = np.nan
                    row["self_percentile"] = np.nan
                    row["self_assessment_gap"] = np.nan
                    row["miscalibration_group"] = "middle"
                    row["batch_trust"] = np.nan
                    row["first_batch_self_assessment_gap"] = first_gap
                    row["first_batch_overestimation"] = int(first_gap > 0)
                    row["first_batch_underestimation"] = int(first_gap < 0)

                row["harmful_reliance"] = int(row["overreliance"] == 1 or row["underreliance"] == 1)
                row["reliance_state"] = (
                    "harmful_ai_overreliance"
                    if row["overreliance"] == 1
                    else "harmful_ai_underreliance"
                    if row["underreliance"] == 1
                    else "beneficial_ai_reliance"
                    if row["beneficial_ai_reliance"] == 1
                    else "correct_self_reliance"
                    if row["correct_self_reliance"] == 1
                    else "independent_correct"
                    if row["initial_correct"] == 1 and row["advice_correct"] == 1 and row["final_correct"] == 1
                    else "independent_incorrect"
                    if row["initial_correct"] == 0 and row["advice_correct"] == 0 and row["final_correct"] == 0
                    else "uncertain_disagreement"
                )

            rows.extend(preview_rows)
            participant_rows.append(
                {
                    "participant_id": participant_id,
                    "dataset_name": "chi2023_dke",
                    "condition_id": participant["condition_id"],
                    "condition_name": participant["condition_name"],
                    "tutorial_present": int(participant["tutorial_present"]),
                    "xai_present": int(participant["xai_present"]),
                    "ati_scale": participant["ati_scale"],
                    "propensity_to_trust": participant["propensity_to_trust"],
                    "trust_first": participant["trust_first"],
                    "trust_second": participant["trust_second"],
                    "xai_helpfulness_normalized": participant["xai_helpfulness_normalized"],
                    "attention_passes": int(participant["attention_passes"]),
                    "first_batch_correct": actual_first_correct,
                    "second_batch_correct": actual_second_correct,
                    "first_batch_self_assessment_gap": first_gap,
                    "second_batch_self_assessment_gap": second_gap,
                    "first_batch_miscalibration_group": first_group,
                    "second_batch_miscalibration_group": second_group,
                    "survey_percentage_first": float(participant["surveyPercentage1"]) / 100.0,
                    "survey_percentage_second": float(participant["surveyPercentage2"]) / 100.0,
                }
            )

        interactions = pd.DataFrame(rows)
        tasks_table = tasks.rename(columns={"answer": "correct_label", "AI-advice": "advice_label", "id_string": "task_instance_key"}).copy()
        tasks_table["task_family"] = "logical_reasoning"
        tasks_table["domain_group"] = "logical_reasoning"
        tasks_table["task_instance_id"] = tasks_table.index

        participant_frame = pd.DataFrame(participant_rows)

        prepared = PreparedRealDataset(
            metadata=self.dataset_card,
            interactions=interactions,
            participants=participant_frame,
            tasks=tasks_table[
                ["task_family", "domain_group", "task_instance_id", "task_instance_key", "correct_label", "advice_label"]
            ],
            prepared_dir=REAL_DATA_PREPARED_DIR / "chi2023_dke",
        )
        prepared.save()
        return prepared
