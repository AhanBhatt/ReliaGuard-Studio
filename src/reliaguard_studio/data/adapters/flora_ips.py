from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from ...paths import REAL_DATA_PREPARED_DIR
from ..real_schema import DatasetCard, PreparedRealDataset
from .registry import BaseRealDatasetAdapter


FIGSHARE_ARTICLE_ID = 32321499
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
CORE_FILES = {
    "ids": "275_student_ids.csv",
    "dialogue": "5.1_dialogue data.xlsx",
    "writing": "5.3_writing_log.xlsx",
    "scores": "5.4_proposal_and_score.xlsx",
    "biographic": "5.5_pre-task_biographic_survey.xlsx",
    "prior": "5.6_pre-prio_knowledge_survey.xlsx",
    "post": "5.7_post-experience_using_platform_survey.xlsx",
    "annotations": "5.8_annotation_log_cleaned.xlsx",
}


def _numeric_prefix(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    token = text.split(":", 1)[0].strip()
    try:
        return float(token)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return float("nan")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _read_xlsx(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required FLoRA file: {path.name}")
    return pd.read_excel(path, **kwargs)


class FloraIPSAdapter(BaseRealDatasetAdapter):
    dataset_card = DatasetCard(
        name="FLoRA GenAI-assisted information problem solving dataset",
        short_name="flora_ips",
        source="Figshare / GitHub",
        source_url="https://doi.org/10.6084/m9.figshare.32321499",
        license_name="CC BY 4.0 on Figshare; companion GitHub repository uses BSD-2-Clause",
        citation="Li X. et al. Dataset of GenAI-Assisted Information Problem Solving in Education. Figshare, 2026.",
        decision="integrated",
        auto_download=True,
        redistributable=True,
        participant_or_interaction_count="275 students with dialogue, writing, annotation, survey and score files",
        domains=["educational writing", "GenAI dialogue", "process tracing"],
        supported_constructs=[
            "process traces",
            "prompt behavior",
            "outcome performance",
            "GenAI intensity",
            "metacognitive engagement",
            "source engagement proxy",
        ],
        note=(
            "Integrated as observational process-trace evidence. The prepared table aggregates dialogue, writing, annotation, "
            "survey and proposal-score files to student-level features; it does not support causal tutoring, delayed-recall or "
            "transfer claims."
        ),
    )

    def _metadata(self) -> dict[str, Any]:
        metadata_path = self.raw_dir / "figshare_metadata.json"
        with httpx.Client(follow_redirects=True, timeout=120.0, headers={"User-Agent": "nsca-research-artifact/0.1"}) as client:
            response = client.get(FIGSHARE_API_URL)
            response.raise_for_status()
        metadata_path.write_text(response.text, encoding="utf-8")
        return response.json()

    def download(self, force: bool = False) -> list[Path]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata()
        outputs = [self.raw_dir / "figshare_metadata.json"]
        with httpx.Client(follow_redirects=True, timeout=300.0, headers={"User-Agent": "nsca-research-artifact/0.1"}) as client:
            for item in metadata.get("files", []):
                name = item.get("name")
                if not name:
                    continue
                destination = self.raw_dir / name
                expected_size = int(item.get("size") or 0)
                if destination.exists() and not force and (expected_size == 0 or destination.stat().st_size == expected_size):
                    outputs.append(destination)
                    continue
                with client.stream("GET", item["download_url"]) as response:
                    response.raise_for_status()
                    tmp = destination.with_suffix(destination.suffix + ".part")
                    with tmp.open("wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                    tmp.replace(destination)
                outputs.append(destination)
        return outputs

    def _dialogue_features(self) -> pd.DataFrame:
        dialogue = _read_xlsx(self.raw_dir / CORE_FILES["dialogue"])
        dialogue["user_id"] = dialogue["user_id"].astype(int)
        code = dialogue["user_utterance_code"].fillna("").astype(str).str.lower()
        chatbot_code = dialogue["chatbot_utterance_code"].fillna("").astype(str).str.lower()
        text = dialogue["user_utterance_text"].fillna("").astype(str)
        dialogue["prompt_word_count"] = text.str.split().str.len()
        dialogue["evaluation_prompt"] = code.str.contains("judgement|revise|preference|interpret").astype(int)
        dialogue["planning_prompt"] = code.str.contains("task-clarification|provide-context|inspire").astype(int)
        dialogue["knowledge_prompt"] = code.str.contains("knowledge").astype(int)
        dialogue["writing_advice_prompt"] = code.str.contains("writing-advice").astype(int)
        dialogue["chitchat_prompt"] = code.str.contains("chitchat|others").astype(int)
        dialogue["chatbot_critique"] = chatbot_code.str.contains("critique").astype(int)
        dialogue["chatbot_no_answer"] = chatbot_code.str.contains("no-answer").astype(int)
        grouped = dialogue.groupby("user_id")
        features = grouped.agg(
            prompt_count=("user_utterance_text", "count"),
            prompt_depth=("prompt_word_count", "mean"),
            prompt_code_diversity=("user_utterance_code", "nunique"),
            evaluation_prompt_count=("evaluation_prompt", "sum"),
            planning_prompt_count=("planning_prompt", "sum"),
            knowledge_prompt_count=("knowledge_prompt", "sum"),
            writing_advice_prompt_count=("writing_advice_prompt", "sum"),
            chitchat_prompt_count=("chitchat_prompt", "sum"),
            chatbot_critique_turns=("chatbot_critique", "sum"),
            chatbot_no_answer_turns=("chatbot_no_answer", "sum"),
        ).reset_index()
        features["metacognitive_prompt_ratio"] = _safe_ratio(
            features["evaluation_prompt_count"] + features["planning_prompt_count"],
            features["prompt_count"],
        )
        features["evaluation_prompt_ratio"] = _safe_ratio(features["evaluation_prompt_count"], features["prompt_count"])
        features["planning_prompt_ratio"] = _safe_ratio(features["planning_prompt_count"], features["prompt_count"])
        features["off_task_rate"] = _safe_ratio(features["chitchat_prompt_count"], features["prompt_count"])
        features["chatbot_critique_ratio"] = _safe_ratio(features["chatbot_critique_turns"], features["prompt_count"])
        return features

    def _writing_features(self) -> pd.DataFrame:
        writing = _read_xlsx(self.raw_dir / CORE_FILES["writing"], usecols=["user_id", "writing_log_time", "writing_content"])
        writing["user_id"] = writing["user_id"].astype(int)
        writing = writing.sort_values(["user_id", "writing_log_time"])
        content = writing["writing_content"].fillna("").astype(str)
        writing["content_length"] = content.str.len()
        writing["word_count"] = content.str.split().str.len()
        writing["length_delta"] = writing.groupby("user_id")["content_length"].diff().fillna(0.0)
        writing["revision_event"] = (writing["length_delta"].abs() > 50).astype(int)
        writing["paste_like_jump"] = (writing["length_delta"] > 500).astype(int)
        writing["deletion_event"] = (writing["length_delta"] < -100).astype(int)
        grouped = writing.groupby("user_id")
        features = grouped.agg(
            writing_event_count=("writing_content", "count"),
            final_word_count=("word_count", "last"),
            max_word_count=("word_count", "max"),
            revision_events=("revision_event", "sum"),
            paste_like_jumps=("paste_like_jump", "sum"),
            deletion_events=("deletion_event", "sum"),
        ).reset_index()
        features["revision_depth"] = _safe_ratio(features["revision_events"], features["writing_event_count"])
        features["copy_paste_write_ratio"] = _safe_ratio(features["paste_like_jumps"], features["writing_event_count"])
        return features

    def _annotation_features(self) -> pd.DataFrame:
        annotations = _read_xlsx(self.raw_dir / CORE_FILES["annotations"])
        annotations["user_id"] = annotations["user_id"].astype(int)
        tag = annotations["default_tag"].fillna("").astype(str).str.lower()
        annotations["important_tag"] = tag.str.contains("important").astype(int)
        annotations["useful_tag"] = tag.str.contains("useful").astype(int)
        annotations["note_tag"] = tag.str.contains("note").astype(int)
        annotations["confusing_tag"] = tag.str.contains("confusing").astype(int)
        grouped = annotations.groupby("user_id")
        return grouped.agg(
            annotation_count=("highlight_text", "count"),
            useful_annotation_count=("useful_tag", "sum"),
            important_annotation_count=("important_tag", "sum"),
            note_annotation_count=("note_tag", "sum"),
            confusing_annotation_count=("confusing_tag", "sum"),
        ).reset_index()

    def _survey_features(self) -> pd.DataFrame:
        biographic = _read_xlsx(self.raw_dir / CORE_FILES["biographic"])
        prior = _read_xlsx(self.raw_dir / CORE_FILES["prior"])
        post = _read_xlsx(self.raw_dir / CORE_FILES["post"])
        for frame in [biographic, prior, post]:
            frame["user_id"] = frame["user_id"].astype(int)
        biographic["age_numeric"] = biographic["age"].apply(_numeric_prefix)
        biographic["degree_level_numeric"] = biographic["degree_level"].apply(_numeric_prefix)
        prior["prior_ds_knowledge"] = prior["has_prior_knowledge_in_DS"].apply(_numeric_prefix)
        prior["prior_genai_knowledge"] = prior["has_prior_knowledge_in_genai"].apply(_numeric_prefix)
        prior["genai_familiarity"] = prior["extent_of_familiar_genai"].apply(_numeric_prefix)
        post["chatbot_topic_usefulness"] = post["extent_chatbot_useful_for_finding_topic"].apply(_numeric_prefix)
        post["chatbot_assignment_usefulness"] = post["extent_chatbot_useful_for_finish_assignment"].apply(_numeric_prefix)
        post["platform_engagement"] = post["extent_engagement"].apply(_numeric_prefix)
        post["subject_understanding_improved"] = post["extent_understanding_of_subject_improved"].apply(_numeric_prefix)
        bio_features = (
            biographic[["user_id", "age_numeric", "degree_level_numeric", "major"]]
            .sort_values("user_id")
            .groupby("user_id", as_index=False)
            .agg({"age_numeric": "mean", "degree_level_numeric": "mean", "major": "first"})
        )
        prior_features = (
            prior[["user_id", "prior_ds_knowledge", "prior_genai_knowledge", "genai_familiarity"]]
            .groupby("user_id", as_index=False)
            .mean(numeric_only=True)
        )
        post_features = (
            post[
                [
                    "user_id",
                    "chatbot_topic_usefulness",
                    "chatbot_assignment_usefulness",
                    "platform_engagement",
                    "subject_understanding_improved",
                ]
            ]
            .groupby("user_id", as_index=False)
            .mean(numeric_only=True)
        )
        return bio_features.merge(prior_features, on="user_id", how="outer").merge(post_features, on="user_id", how="outer")

    def prepare(self) -> PreparedRealDataset:
        missing = [name for name in CORE_FILES.values() if not (self.raw_dir / name).exists()]
        if missing:
            self.download(force=False)
        ids = pd.read_csv(self.raw_dir / CORE_FILES["ids"])
        ids["user_id"] = ids["user_id"].astype(int)
        scores = _read_xlsx(self.raw_dir / CORE_FILES["scores"])
        scores["user_id"] = scores["user_id"].astype(int)
        scores["proposal_score"] = pd.to_numeric(scores["proposal_score"], errors="coerce")
        max_score = float(scores["proposal_score"].max() or 15.0)
        scores["proposal_score_normalized"] = scores["proposal_score"] / max_score

        features = (
            ids.merge(scores[["user_id", "proposal_score", "proposal_score_normalized"]], on="user_id", how="left")
            .merge(self._dialogue_features(), on="user_id", how="left")
            .merge(self._writing_features(), on="user_id", how="left")
            .merge(self._annotation_features(), on="user_id", how="left")
            .merge(self._survey_features(), on="user_id", how="left")
        )
        count_columns = [
            "prompt_count",
            "prompt_depth",
            "prompt_code_diversity",
            "evaluation_prompt_count",
            "planning_prompt_count",
            "knowledge_prompt_count",
            "writing_advice_prompt_count",
            "chitchat_prompt_count",
            "chatbot_critique_turns",
            "chatbot_no_answer_turns",
            "metacognitive_prompt_ratio",
            "evaluation_prompt_ratio",
            "planning_prompt_ratio",
            "off_task_rate",
            "chatbot_critique_ratio",
            "writing_event_count",
            "final_word_count",
            "max_word_count",
            "revision_events",
            "paste_like_jumps",
            "deletion_events",
            "revision_depth",
            "copy_paste_write_ratio",
            "annotation_count",
            "useful_annotation_count",
            "important_annotation_count",
            "note_annotation_count",
            "confusing_annotation_count",
        ]
        for column in count_columns:
            if column in features:
                features[column] = features[column].fillna(0.0)

        features["student_id"] = features["user_id"].astype(str)
        features["participant_id"] = "flora_" + features["student_id"]
        features["dataset_name"] = self.dataset_card.short_name
        features["task_family"] = "genai_assisted_information_problem_solving"
        features["task_instance_key"] = "project_proposal"
        features["condition_name"] = "observational_process_trace"
        features["condition_id"] = "observational_process_trace"
        features["genai_intensity"] = _safe_ratio(
            features["prompt_count"],
            features["prompt_count"] + features["writing_event_count"] + features["annotation_count"],
        )
        features["source_engagement_proxy"] = _safe_ratio(features["annotation_count"], features["prompt_count"] + features["annotation_count"])
        features["reading_to_chatbot_balance"] = _safe_ratio(features["annotation_count"], features["prompt_count"])
        features["annotation_to_chatbot_balance"] = features["reading_to_chatbot_balance"]
        features["process_diversity"] = (
            (features["prompt_count"] > 0).astype(int)
            + (features["writing_event_count"] > 0).astype(int)
            + (features["annotation_count"] > 0).astype(int)
            + (features["revision_events"] > 0).astype(int)
            + (features["chatbot_critique_turns"] > 0).astype(int)
        )
        median_score = float(features["proposal_score_normalized"].median())
        features["outcome_score"] = features["proposal_score_normalized"]
        features["high_performance"] = (features["proposal_score_normalized"] >= median_score).astype(int)
        features["reliance_state"] = np.where(
            (features["high_performance"] == 1) & (features["metacognitive_prompt_ratio"] >= features["metacognitive_prompt_ratio"].median()),
            "high_score_metacognitive_process",
            np.where(
                (features["high_performance"] == 0) & (features["copy_paste_write_ratio"] >= features["copy_paste_write_ratio"].median()),
                "low_score_paste_heavy_process",
                "observational_process_trace",
            ),
        )
        features["temporal_regulation_profile"] = np.where(
            features["process_diversity"] >= 4,
            "diverse_process",
            np.where(features["prompt_count"] > features["writing_event_count"], "chatbot_dominant", "writing_dominant"),
        )
        features["appropriate_reliance"] = pd.NA
        features["overreliance"] = pd.NA
        features["underreliance"] = pd.NA
        features["disagreement_case"] = 0

        interactions = features.copy()
        participants = features[
            [
                "participant_id",
                "student_id",
                "proposal_score",
                "proposal_score_normalized",
                "prompt_count",
                "writing_event_count",
                "annotation_count",
                "metacognitive_prompt_ratio",
                "evaluation_prompt_ratio",
                "revision_depth",
                "source_engagement_proxy",
                "prior_ds_knowledge",
                "prior_genai_knowledge",
                "genai_familiarity",
                "subject_understanding_improved",
            ]
        ].copy()
        tasks = pd.DataFrame(
            [
                {
                    "task_instance_key": "project_proposal",
                    "task_family": "genai_assisted_information_problem_solving",
                    "description": "Student project proposal with GenAI dialogue, writing and annotation process traces.",
                }
            ]
        )
        prepared = PreparedRealDataset(
            metadata=self.dataset_card,
            interactions=interactions,
            participants=participants,
            tasks=tasks,
            prepared_dir=REAL_DATA_PREPARED_DIR / "flora_ips",
            extra_tables={"download_metadata": pd.DataFrame(json.loads((self.raw_dir / "figshare_metadata.json").read_text(encoding="utf-8")).get("files", []))},
        )
        prepared.save()
        return prepared
