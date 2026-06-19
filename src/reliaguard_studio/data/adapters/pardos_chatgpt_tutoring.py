from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from ...paths import REAL_DATA_PREPARED_DIR
from ..real_schema import DatasetCard, PreparedRealDataset
from .registry import BaseRealDatasetAdapter


FIGSHARE_ARTICLE_ID = 23935269
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"


class PardosChatGPTTutoringAdapter(BaseRealDatasetAdapter):
    dataset_card = DatasetCard(
        name="Pardos and Bhandari ChatGPT tutoring dataset",
        short_name="pardos_chatgpt_tutoring",
        source="Figshare",
        source_url="https://doi.org/10.6084/m9.figshare.23935269",
        license_name="CC BY 4.0",
        citation=(
            "Pardos Z.A., Bhandari S. ChatGPT-generated help produces learning gains equivalent "
            "to human tutor-authored help on mathematics skills. PLOS ONE 2024."
        ),
        decision="integrated",
        auto_download=True,
        redistributable=True,
        participant_or_interaction_count="274 participants",
        domains=["mathematics tutoring"],
        supported_constructs=["learning gain", "condition effects", "time on task", "outcome performance"],
        note="Public Figshare CC BY 4.0 participant-level file; integrated as learning-gain evidence, not reliance evidence.",
    )

    def download(self, force: bool = False) -> list[Path]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        target = self.raw_dir / "Participants.csv"
        metadata_path = self.raw_dir / "figshare_metadata.json"
        if target.exists() and metadata_path.exists() and not force:
            return [target, metadata_path]

        with httpx.Client(follow_redirects=True, timeout=120.0, headers={"User-Agent": "nsca-research-artifact/0.1"}) as client:
            metadata = client.get(FIGSHARE_API_URL)
            metadata.raise_for_status()
            metadata_path.write_text(metadata.text, encoding="utf-8")
            files = metadata.json().get("files", [])
            participant_file = next((item for item in files if item.get("name", "").lower() == "participants.csv"), None)
            if participant_file is None:
                raise FileNotFoundError("Figshare metadata did not list Participants.csv for the Pardos tutoring dataset.")
            response = client.get(participant_file["download_url"])
            response.raise_for_status()
            target.write_bytes(response.content)
        return [target, metadata_path]

    def prepare(self) -> PreparedRealDataset:
        self.download(force=False)
        raw_path = self.raw_dir / "Participants.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                "Participants.csv not found. Manual download is required for the Pardos tutoring dataset."
            )
        frame = pd.read_csv(raw_path)
        normalized = frame.rename(
            columns={
                "anonUserID": "participant_id",
                "lesson": "math_topic",
                "condition": "condition_name",
                "preTest": "pre_test_score",
                "postTest": "post_test_score",
                "sessionTime": "session_time_seconds",
                "totalUniqueSteps": "total_unique_steps",
                "learningGain": "learning_gain",
            }
        ).copy()
        normalized["dataset_name"] = self.dataset_card.short_name
        normalized["task_family"] = "mathematics_tutoring"
        normalized["task_instance_key"] = normalized["math_topic"].astype(str)
        normalized["condition_id"] = (
            normalized["condition_name"]
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )
        normalized["help_condition"] = normalized["condition_id"]
        normalized["chatgpt_help"] = (normalized["condition_id"] == "chatgpt").astype(int)
        normalized["human_tutor_help"] = (normalized["condition_id"] == "human_tutor").astype(int)
        normalized["no_help_control"] = (normalized["condition_id"] == "no_hint_control").astype(int)
        normalized["initial_correct"] = normalized["pre_test_score"].astype(float)
        normalized["final_correct"] = normalized["post_test_score"].astype(float)
        normalized["learning_gain_positive"] = (normalized["learning_gain"] > 0).astype(int)
        normalized["reliance_state"] = normalized["condition_id"].map(
            {
                "chatgpt": "chatgpt_help_learning",
                "human_tutor": "human_tutor_learning",
                "no_hint_control": "no_help_learning",
            }
        ).fillna("learning_record")
        normalized["appropriate_reliance"] = pd.NA
        normalized["overreliance"] = pd.NA
        normalized["underreliance"] = pd.NA
        normalized["disagreement_case"] = 0
        participants = normalized[
            [
                "participant_id",
                "condition_name",
                "math_topic",
                "pre_test_score",
                "post_test_score",
                "learning_gain",
                "session_time_seconds",
                "total_unique_steps",
            ]
        ].copy()
        tasks = normalized[["task_instance_key", "task_family", "math_topic"]].drop_duplicates().reset_index(drop=True)
        prepared = PreparedRealDataset(
            metadata=self.dataset_card,
            interactions=normalized,
            participants=participants,
            tasks=tasks,
            prepared_dir=REAL_DATA_PREPARED_DIR / "pardos_chatgpt_tutoring",
        )
        prepared.save()
        return prepared
