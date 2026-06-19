from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GEESpec:
    dataset: str
    outcome: str
    family: str
    formula: str
    subset_query: str | None = None
    note: str = ""


def _clean_for_formula(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in columns:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
    return cleaned.dropna(subset=columns)


def _fit_gee(frame: pd.DataFrame, spec: GEESpec) -> list[dict[str, Any]]:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    analysis = frame.query(spec.subset_query).copy() if spec.subset_query else frame.copy()
    if analysis.empty or analysis[spec.outcome].nunique(dropna=True) < 2:
        return [
            {
                "dataset": spec.dataset,
                "outcome": spec.outcome,
                "term": "model_not_fit",
                "estimate": np.nan,
                "std_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "n": int(len(analysis)),
                "clusters": 0,
                "family": spec.family,
                "formula": spec.formula,
                "note": "insufficient outcome variation",
            }
        ]

    analysis["participant_cluster"] = analysis["participant_id"].astype(str)
    if spec.family == "binomial":
        family = sm.families.Binomial()
    elif spec.family == "gaussian":
        family = sm.families.Gaussian()
    else:
        raise ValueError(f"Unsupported GEE family: {spec.family}")

    try:
        model = smf.gee(spec.formula, groups="participant_cluster", data=analysis, family=family)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            result = model.fit(maxiter=80)
    except Exception as exc:
        return [
            {
                "dataset": spec.dataset,
                "outcome": spec.outcome,
                "term": "model_not_fit",
                "estimate": np.nan,
                "std_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "n": int(len(analysis)),
                "clusters": int(analysis["participant_cluster"].nunique()),
                "family": spec.family,
                "formula": spec.formula,
                "note": f"GEE failed: {type(exc).__name__}: {exc}",
            }
        ]

    rows = []
    with warnings.catch_warnings(record=True) as post_fit_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        conf = result.conf_int()
        bse = result.bse
        pvalues = result.pvalues
    warning_note = "; ".join(sorted({str(warning.message) for warning in [*caught_warnings, *post_fit_warnings]}))
    for term, estimate in result.params.items():
        term_note = spec.note or "GEE with participant clusters and robust covariance"
        if warning_note and (
            not np.isfinite(bse[term])
            or not np.isfinite(conf.loc[term, 0])
            or not np.isfinite(conf.loc[term, 1])
        ):
            term_note = f"{term_note} Robust covariance warning for this term: {warning_note}"
        rows.append(
            {
                "dataset": spec.dataset,
                "outcome": spec.outcome,
                "term": term,
                "estimate": float(estimate),
                "std_error": float(bse[term]),
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "p_value": float(pvalues[term]),
                "n": int(result.nobs),
                "clusters": int(analysis["participant_cluster"].nunique()),
                "family": spec.family,
                "formula": spec.formula,
                "note": term_note,
            }
        )
    return rows


def run_gee_models(interactions_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    specs = [
        GEESpec(
            dataset="haiid",
            outcome="overreliance",
            family="binomial",
            subset_query="initial_correct == 1 and advice_correct == 0",
            formula="overreliance ~ C(advice_source_label) + C(task_family) + initial_confidence + stated_accuracy_normalized",
            note="Overreliance modeled only for initially correct participants receiving wrong advice.",
        ),
        GEESpec(
            dataset="haiid",
            outcome="underreliance",
            family="binomial",
            subset_query="initial_correct == 0 and advice_correct == 1",
            formula="underreliance ~ C(advice_source_label) + C(task_family) + initial_confidence + stated_accuracy_normalized",
            note="Underreliance modeled only for initially wrong participants receiving correct advice.",
        ),
        GEESpec(
            dataset="chi2023_dke",
            outcome="appropriate_reliance",
            family="binomial",
            subset_query="analysis_included == 1 and analysis_batch == 'second' and disagreement_case == 1",
            formula="appropriate_reliance ~ tutorial_present + xai_present + first_batch_overestimation + first_batch_underestimation + trust_first + ati_scale + propensity_to_trust",
            note="Appropriate reliance among second-batch disagreement cases.",
        ),
        GEESpec(
            dataset="convxai_iui2025",
            outcome="appropriate_reliance",
            family="binomial",
            subset_query="disagreement_case == 1",
            formula="appropriate_reliance ~ C(condition_id) + initial_confidence + confidence_change + post_explain_reliability + user_question_rate + chat_turn_count",
            note="Appropriate reliance in ConvXAI disagreement cases.",
        ),
        GEESpec(
            dataset="convxai_iui2025",
            outcome="overreliance",
            family="binomial",
            subset_query="initial_correct == 1 and advice_correct == 0",
            formula="overreliance ~ C(condition_id) + initial_confidence + post_explain_reliability + user_question_rate + chat_turn_count",
            note="Overreliance among initially correct ConvXAI participants receiving wrong model advice.",
        ),
        GEESpec(
            dataset="pardos_chatgpt_tutoring",
            outcome="learning_gain",
            family="gaussian",
            formula="learning_gain ~ C(condition_name) + C(math_topic) + pre_test_score + session_time_seconds + total_unique_steps",
            note="Learning gain modeled with participant clusters; each participant contributes one record.",
        ),
        GEESpec(
            dataset="flora_ips",
            outcome="proposal_score_normalized",
            family="gaussian",
            formula=(
                "proposal_score_normalized ~ genai_intensity + metacognitive_prompt_ratio + "
                "source_engagement_proxy + revision_depth + copy_paste_write_ratio + prior_ds_knowledge + genai_familiarity"
            ),
            note="Proposal score modeled as an observational association with student-level process traces; not causal.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.dataset not in interactions_map:
            continue
        rows.extend(_fit_gee(interactions_map[spec.dataset], spec))
    return pd.DataFrame(rows)
