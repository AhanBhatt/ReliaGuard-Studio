from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...paths import REAL_DATA_PREPARED_DIR
from ..real_schema import DatasetCard, PreparedRealDataset
from .registry import BaseRealDatasetAdapter, download_file


RAW_FILES = {
    "haiid_dataset.csv": "https://raw.githubusercontent.com/kailas-v/human-ai-interactions/main/haiid_dataset.csv",
    "haiid_dataset_description.csv": "https://raw.githubusercontent.com/kailas-v/human-ai-interactions/main/haiid_dataset_description.csv",
    "haiid.py": "https://raw.githubusercontent.com/kailas-v/human-ai-interactions/main/haiid.py",
    "README.md": "https://raw.githubusercontent.com/kailas-v/human-ai-interactions/main/README.md",
    "LICENSE.txt": "https://raw.githubusercontent.com/kailas-v/human-ai-interactions/main/LICENSE.txt",
}


class HAIIDAdapter(BaseRealDatasetAdapter):
    dataset_card = DatasetCard(
        name="Human-AI Interactions Dataset",
        short_name="haiid",
        source="GitHub",
        source_url="https://github.com/kailas-v/human-ai-interactions",
        license_name="MIT",
        citation=(
            "Vodrahalli K., Daneshjou R., Gerstenberg T., Zou J. "
            "Do humans trust advice more if it comes from AI? An analysis of human-AI interactions. AIES 2022."
        ),
        decision="integrated",
        auto_download=True,
        redistributable=True,
        participant_or_interaction_count="35,670 interactions",
        domains=["art", "census", "cities", "dermatology", "sarcasm"],
        supported_constructs=[
            "appropriate reliance",
            "overreliance",
            "underreliance",
            "confidence shift",
            "team improvement",
            "advice source effects",
        ],
        note="Core files can be downloaded directly even though a full Windows checkout of the repo fails on an invalid task path.",
    )

    def download(self, force: bool = False) -> list[Path]:
        paths = []
        for filename, url in RAW_FILES.items():
            paths.append(download_file(url, self.raw_dir / filename, force=force))
        return paths

    def prepare(self) -> PreparedRealDataset:
        self.download(force=False)
        frame = pd.read_csv(self.raw_dir / "haiid_dataset.csv", low_memory=False)
        frame = frame.copy()
        frame["dataset_name"] = "haiid"
        frame["participant_id"] = frame["participant_id"].astype(str)
        frame["task_family"] = frame["task_name"]
        frame["domain_group"] = frame["task_name"]
        frame["task_instance_key"] = frame["task_name"].astype(str) + "::" + frame["task_instance_id"].astype(str)
        frame["advice_source_label"] = frame["advice_source"].astype(str)
        frame["advice_source_ai"] = (frame["advice_source"].astype(str).str.lower() == "ai").astype(int)
        frame["initial_response_value"] = frame["response_1"].astype(float)
        frame["final_response_value"] = frame["response_2"].astype(float)
        frame["advice_value"] = frame["advice"].astype(float)
        frame["initial_confidence"] = frame["initial_response_value"].abs()
        frame["final_confidence"] = frame["final_response_value"].abs()
        frame["confidence_change"] = frame["final_confidence"] - frame["initial_confidence"]
        frame["initial_correct"] = (frame["initial_response_value"] > 0).astype(int)
        frame["initial_incorrect"] = (frame["initial_response_value"] < 0).astype(int)
        frame["final_correct"] = (frame["final_response_value"] > 0).astype(int)
        frame["advice_correct"] = (frame["advice_value"] > 0).astype(int)
        frame["advice_wrong"] = 1 - frame["advice_correct"]
        frame["initial_label"] = np.where(frame["initial_response_value"] >= 0, frame["correct_label"], frame["incorrect_label"])
        frame["final_label"] = np.where(frame["final_response_value"] >= 0, frame["correct_label"], frame["incorrect_label"])
        frame["advice_label"] = np.where(frame["advice_value"] >= 0, frame["correct_label"], frame["incorrect_label"])
        frame["disagreement_case"] = (np.sign(frame["initial_response_value"]) != np.sign(frame["advice_value"])).astype(int)
        frame["switch_behavior"] = (np.sign(frame["initial_response_value"]) != np.sign(frame["final_response_value"])).astype(int)
        frame["movement_toward_advice"] = (
            np.abs(frame["final_response_value"] - frame["advice_value"]) + 1e-6
            < np.abs(frame["initial_response_value"] - frame["advice_value"])
        ).astype(int)
        frame["movement_away_from_advice"] = (
            np.abs(frame["final_response_value"] - frame["advice_value"])
            > np.abs(frame["initial_response_value"] - frame["advice_value"]) + 1e-6
        ).astype(int)
        frame["advice_uptake"] = (
            (np.sign(frame["final_response_value"]) == np.sign(frame["advice_value"]))
            & (np.sign(frame["initial_response_value"]) != np.sign(frame["advice_value"]))
        ).astype(int)
        frame["human_team_improvement"] = frame["final_correct"] - frame["initial_correct"]
        frame["correct_ai_reliance"] = (
            (frame["initial_correct"] == 0)
            & (frame["advice_correct"] == 1)
            & (frame["final_correct"] == 1)
        ).astype(int)
        frame["correct_self_reliance"] = (
            (frame["initial_correct"] == 1)
            & (frame["advice_correct"] == 0)
            & (frame["final_correct"] == 1)
        ).astype(int)
        frame["overreliance"] = (
            (frame["initial_correct"] == 1)
            & (frame["advice_correct"] == 0)
            & (frame["final_correct"] == 0)
            & (frame["advice_uptake"] == 1)
        ).astype(int)
        frame["underreliance"] = (
            (frame["initial_correct"] == 0)
            & (frame["advice_correct"] == 1)
            & (frame["final_correct"] == 0)
        ).astype(int)
        frame["beneficial_ai_reliance"] = frame["correct_ai_reliance"]
        frame["harmful_reliance"] = ((frame["overreliance"] == 1) | (frame["underreliance"] == 1)).astype(int)
        frame["appropriate_reliance"] = np.where(frame["disagreement_case"] == 1, frame["final_correct"], np.nan)
        frame["confidence_inflated_reliance"] = (
            (frame["movement_toward_advice"] == 1) & (frame["confidence_change"] > 0.10)
        ).astype(int)
        frame["stated_accuracy_normalized"] = frame["perceived_accuracy"].astype(float) / 100.0
        frame["perceived_helpfulness"] = frame["survey_q2_helpfulness_of_advice"].astype(float)
        frame["trust_in_advice"] = frame["survey_q3_trust_in_advice"].astype(float)
        frame["ai_usage_frequency"] = frame["survey_q5_AI_in_life"].astype(float)
        frame["education_level_numeric"] = pd.to_numeric(frame["education"], errors="coerce")
        frame["programming_experience_numeric"] = frame["programming_experience"].astype(str).str.lower().isin({"true", "1"}).astype(int)
        frame["expert_years"] = pd.to_numeric(frame["years_of_experience"], errors="coerce").fillna(0.0)
        frame["age_numeric"] = pd.to_numeric(frame["age"], errors="coerce")
        frame["socioeconomic_status_numeric"] = pd.to_numeric(frame["socioeconomic_status"], errors="coerce")
        task_risk = frame.groupby("task_family")["overreliance"].mean().rename("task_family_overreliance_rate")
        frame = frame.merge(task_risk, on="task_family", how="left")

        state = np.select(
            [
                frame["overreliance"] == 1,
                frame["underreliance"] == 1,
                frame["correct_ai_reliance"] == 1,
                frame["correct_self_reliance"] == 1,
                (frame["initial_correct"] == 1) & (frame["advice_correct"] == 1) & (frame["final_correct"] == 1),
                (frame["initial_correct"] == 0) & (frame["advice_correct"] == 0) & (frame["final_correct"] == 0),
                (frame["movement_toward_advice"] == 1) & (frame["confidence_change"] > 0.10),
            ],
            [
                "harmful_ai_overreliance",
                "harmful_ai_underreliance",
                "beneficial_ai_reliance",
                "correct_self_reliance",
                "independent_correct",
                "independent_incorrect",
                "confidence_inflated_reliance",
            ],
            default="uncertain_disagreement",
        )
        frame["reliance_state"] = state

        interactions = frame[
            [
                "dataset_name",
                "participant_id",
                "task_family",
                "domain_group",
                "task_instance_id",
                "task_instance_key",
                "advice_source_label",
                "advice_source_ai",
                "perceived_accuracy",
                "stated_accuracy_normalized",
                "initial_response_value",
                "final_response_value",
                "advice_value",
                "correct_label",
                "incorrect_label",
                "initial_label",
                "final_label",
                "advice_label",
                "initial_correct",
                "final_correct",
                "advice_correct",
                "advice_wrong",
                "initial_confidence",
                "final_confidence",
                "confidence_change",
                "disagreement_case",
                "switch_behavior",
                "advice_uptake",
                "movement_toward_advice",
                "movement_away_from_advice",
                "human_team_improvement",
                "correct_ai_reliance",
                "correct_self_reliance",
                "overreliance",
                "underreliance",
                "beneficial_ai_reliance",
                "harmful_reliance",
                "appropriate_reliance",
                "confidence_inflated_reliance",
                "reliance_state",
                "task_family_overreliance_rate",
                "perceived_helpfulness",
                "trust_in_advice",
                "ai_usage_frequency",
                "geographic_region",
                "education_level_numeric",
                "education_description",
                "gender",
                "age_numeric",
                "programming_experience_numeric",
                "socioeconomic_status_numeric",
                "expert_years",
                "job_title",
            ]
        ].copy()

        participants = (
            interactions.groupby("participant_id")
            .agg(
                dataset_name=("dataset_name", "first"),
                mean_initial_accuracy=("initial_correct", "mean"),
                mean_final_accuracy=("final_correct", "mean"),
                overreliance_rate=("overreliance", "mean"),
                underreliance_rate=("underreliance", "mean"),
                appropriate_reliance_rate=("appropriate_reliance", "mean"),
                mean_initial_confidence=("initial_confidence", "mean"),
                mean_trust_in_advice=("trust_in_advice", "mean"),
                mean_ai_usage_frequency=("ai_usage_frequency", "mean"),
                age_numeric=("age_numeric", "mean"),
                education_level_numeric=("education_level_numeric", "mean"),
                programming_experience_numeric=("programming_experience_numeric", "max"),
                expert_years=("expert_years", "max"),
            )
            .reset_index()
        )

        tasks = (
            frame[
                [
                    "task_family",
                    "domain_group",
                    "task_instance_id",
                    "task_instance_key",
                    "correct_label",
                    "incorrect_label",
                    "advice_source_label",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        prepared = PreparedRealDataset(
            metadata=self.dataset_card,
            interactions=interactions,
            participants=participants,
            tasks=tasks,
            prepared_dir=REAL_DATA_PREPARED_DIR / "haiid",
        )
        prepared.save()
        return prepared
