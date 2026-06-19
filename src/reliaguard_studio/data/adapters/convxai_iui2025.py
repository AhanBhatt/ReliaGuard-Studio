from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...paths import REAL_DATA_PREPARED_DIR
from ..real_schema import DatasetCard, PreparedRealDataset
from .registry import BaseRealDatasetAdapter, download_file


RAW_FILES = {
    "README.md": "https://raw.githubusercontent.com/delftcrowd/IUI2025_ConvXAI/main/README.md",
    "task_details.csv": "https://raw.githubusercontent.com/delftcrowd/IUI2025_ConvXAI/main/task_details.csv",
    "xailabdata_all.csv": "https://raw.githubusercontent.com/delftcrowd/IUI2025_ConvXAI/main/data/xailabdata_all.csv",
    "xailabdata_llm_agent.csv": "https://raw.githubusercontent.com/delftcrowd/IUI2025_ConvXAI/main/data/xailabdata_llm_agent.csv",
}


def _loads(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _bool01(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1.0
    if text in {"false", "0", "no", "n"}:
        return 0.0
    return np.nan


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _engagement_features(payload: dict[str, Any]) -> dict[str, float]:
    explainer = payload.get("explainerData", {})
    if not isinstance(explainer, dict):
        explainer = {}
    metadata = explainer.get("chatSessionMetadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    chat_history = explainer.get("chat_history", [])
    if not isinstance(chat_history, list):
        chat_history = []
    journey = explainer.get("chatJourney", [])
    if not isinstance(journey, list):
        journey = []
    user_turns = sum(1 for item in chat_history if isinstance(item, str) and item.strip().startswith("[user]"))
    return {
        "api_calls": _numeric(explainer.get("api_calls", 0.0)),
        "chat_turn_count": float(len(chat_history)),
        "user_question_count": float(user_turns),
        "user_question_rate": min(float(user_turns) / 5.0, 1.0),
        "journey_step_count": float(len(journey)),
        "pdp_count": _numeric(metadata.get("pdp", 0.0)),
        "shap_count": _numeric(metadata.get("shap", 0.0)),
        "whatif_count": _numeric(metadata.get("whatIf", 0.0)),
        "counterfactual_count": _numeric(metadata.get("counterFactual", 0.0)),
        "decision_tree_count": _numeric(metadata.get("decisionTree", 0.0)),
    }


class ConvXAIIUI2025Adapter(BaseRealDatasetAdapter):
    dataset_card = DatasetCard(
        name="IUI 2025 Conversational XAI Human-AI Decision-Making Dataset",
        short_name="convxai_iui2025",
        source="GitHub and Research Software Directory",
        source_url="https://github.com/delftcrowd/IUI2025_ConvXAI",
        license_name="CC-BY-4.0 according to the Research Software Directory entry",
        citation=(
            "He G., Aishwarya N., Gadiraju U. Is Conversational XAI All You Need? "
            "Human-AI Decision Making With a Conversational XAI Assistant. IUI 2025."
        ),
        decision="integrated",
        auto_download=True,
        redistributable=True,
        participant_or_interaction_count="306 participants, 3,060 task decisions after combining base and LLM-agent releases",
        domains=["loan approval", "explainable AI", "conversational XAI"],
        supported_constructs=[
            "appropriate reliance",
            "overreliance",
            "underreliance",
            "confidence shift",
            "process traces",
            "verification proxy",
            "outcome performance",
        ],
        note=(
            "Provides pre/post decisions, confidence, model prediction, ground truth, XAI condition, "
            "and conversational interaction traces. It supports decision reliance, not delayed recall or transfer."
        ),
    )

    def download(self, force: bool = False) -> list[Path]:
        paths = []
        data_dir = self.raw_dir / "data"
        for filename, url in RAW_FILES.items():
            target = data_dir / filename if filename.startswith("xailabdata_") else self.raw_dir / filename
            paths.append(download_file(url, target, force=force))
        return paths

    def prepare(self) -> PreparedRealDataset:
        self.download(force=False)
        base = pd.read_csv(self.raw_dir / "data" / "xailabdata_all.csv")
        llm = pd.read_csv(self.raw_dir / "data" / "xailabdata_llm_agent.csv")
        combined = pd.concat([base, llm.drop(columns=["id"], errors="ignore")], ignore_index=True)
        combined["payload"] = combined["data"].map(_loads)
        tasks = pd.read_csv(self.raw_dir / "task_details.csv")
        tasks["advice_value"] = tasks["Model Prediction"].map(_bool01)
        tasks["correct_label"] = tasks["Ground Truth"].map(_bool01)
        task_lookup = tasks.set_index("Taskid").to_dict("index")

        payload_lookup = {
            (str(row.userid), str(row.group), str(row.page)): row.payload
            for row in combined[["userid", "group", "page", "payload"]].itertuples(index=False)
        }
        users = combined[["userid", "group"]].drop_duplicates()
        rows: list[dict[str, Any]] = []
        for user in users.itertuples(index=False):
            participant_id = str(user.userid)
            group = str(user.group)
            for task_id, task_info in task_lookup.items():
                pre = payload_lookup.get((participant_id, group, f"{task_id}_pre"), {})
                post = payload_lookup.get((participant_id, group, f"{task_id}_post"), {})
                if not pre or not post:
                    continue
                initial = _bool01(pre.get("beforeExplainDecision"))
                final = _bool01(post.get("postExplainDecision"))
                advice = float(task_info["advice_value"])
                correct = float(task_info["correct_label"])
                if np.isnan(initial) or np.isnan(final) or np.isnan(advice) or np.isnan(correct):
                    continue

                initial_correct = int(initial == correct)
                final_correct = int(final == correct)
                advice_correct = int(advice == correct)
                disagreement = int(initial != advice)
                switch = int(initial != final)
                advice_uptake = int(disagreement == 1 and final == advice)
                correct_ai_reliance = int(initial_correct == 0 and advice_correct == 1 and final_correct == 1)
                correct_self_reliance = int(initial_correct == 1 and advice_correct == 0 and final_correct == 1)
                overreliance = int(initial_correct == 1 and advice_correct == 0 and final == advice)
                underreliance = int(initial_correct == 0 and advice_correct == 1 and final != advice)
                engagement = _engagement_features(post)
                condition_name = {
                    "control": "control",
                    "dashboard": "xai dashboard",
                    "chatxai": "conversational xai",
                    "chatxaiboost": "conversational xai + boost",
                    "chatxaiAuto": "llm-agent conversational xai",
                }.get(group, group)
                confidence_before = _numeric(pre.get("beforeExplainConfidence")) / 5.0
                confidence_after = _numeric(post.get("postExplainConfidence")) / 5.0
                reliability_after = _numeric(post.get("postExplainReliability")) / 5.0
                xai_present = int(group != "control")
                conversational_xai = int("chat" in group.lower())
                llm_agent = int(group.lower() == "chatxaiauto")
                intervention_responsive = int(correct_ai_reliance == 1 and xai_present == 1)
                state = "uncertain_disagreement"
                if overreliance:
                    state = "harmful_ai_overreliance"
                elif underreliance:
                    state = "harmful_ai_underreliance"
                elif correct_ai_reliance:
                    state = "beneficial_ai_reliance"
                elif correct_self_reliance:
                    state = "correct_self_reliance"
                elif initial_correct and final_correct:
                    state = "independent_correct"
                elif not initial_correct and not final_correct:
                    state = "independent_incorrect"
                elif intervention_responsive:
                    state = "intervention_responsive_reliance"
                elif confidence_after > confidence_before and advice_uptake:
                    state = "confidence_inflated_reliance"

                rows.append(
                    {
                        "dataset_name": "convxai_iui2025",
                        "participant_id": participant_id,
                        "task_family": "loan_creditworthiness",
                        "domain_group": "loan_approval",
                        "task_instance_id": task_id,
                        "task_instance_key": f"loan_approval::{task_id}",
                        "condition_id": group,
                        "condition_name": condition_name,
                        "advice_source_label": "model_prediction",
                        "advice_source_ai": 1,
                        "xai_present": xai_present,
                        "conversational_xai": conversational_xai,
                        "llm_agent": llm_agent,
                        "initial_response_value": initial,
                        "final_response_value": final,
                        "advice_value": advice,
                        "correct_label": correct,
                        "initial_correct": initial_correct,
                        "final_correct": final_correct,
                        "advice_correct": advice_correct,
                        "advice_wrong": 1 - advice_correct,
                        "initial_incorrect": 1 - initial_correct,
                        "initial_confidence": confidence_before,
                        "final_confidence": confidence_after,
                        "confidence_change": confidence_after - confidence_before,
                        "post_explain_reliability": reliability_after,
                        "disagreement_case": disagreement,
                        "switch_behavior": switch,
                        "advice_uptake": advice_uptake,
                        "movement_toward_advice": advice_uptake,
                        "movement_away_from_advice": int(disagreement == 1 and final == initial),
                        "human_team_improvement": final_correct - initial_correct,
                        "correct_ai_reliance": correct_ai_reliance,
                        "correct_self_reliance": correct_self_reliance,
                        "overreliance": overreliance,
                        "underreliance": underreliance,
                        "beneficial_ai_reliance": correct_ai_reliance,
                        "harmful_reliance": int(overreliance == 1 or underreliance == 1),
                        "appropriate_reliance": float(final_correct) if disagreement else np.nan,
                        "confidence_inflated_reliance": int(confidence_after > confidence_before and advice_uptake),
                        "intervention_responsive_reliance": intervention_responsive,
                        "reliance_state": state,
                        "task_position_normalized": int(str(task_id).replace("task", "")) / 10.0,
                        **engagement,
                    }
                )

        interactions = pd.DataFrame(rows)
        participants = (
            interactions.groupby("participant_id")
            .agg(
                dataset_name=("dataset_name", "first"),
                condition_id=("condition_id", "first"),
                condition_name=("condition_name", "first"),
                mean_initial_accuracy=("initial_correct", "mean"),
                mean_final_accuracy=("final_correct", "mean"),
                overreliance_rate=("overreliance", "mean"),
                underreliance_rate=("underreliance", "mean"),
                appropriate_reliance_rate=("appropriate_reliance", "mean"),
                mean_initial_confidence=("initial_confidence", "mean"),
                mean_final_confidence=("final_confidence", "mean"),
                mean_user_questions=("user_question_count", "mean"),
            )
            .reset_index()
        )
        task_table = tasks.rename(columns={"Taskid": "task_instance_id", "Dataset ID": "source_instance_id"}).copy()
        task_table["dataset_name"] = "convxai_iui2025"
        task_table["task_family"] = "loan_creditworthiness"
        task_table["domain_group"] = "loan_approval"
        task_table["task_instance_key"] = "loan_approval::" + task_table["task_instance_id"].astype(str)

        prepared = PreparedRealDataset(
            metadata=self.dataset_card,
            interactions=interactions,
            participants=participants,
            tasks=task_table,
            prepared_dir=REAL_DATA_PREPARED_DIR / "convxai_iui2025",
        )
        prepared.save()
        return prepared
