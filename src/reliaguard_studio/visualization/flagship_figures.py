from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, ensure_directories
from .figure_canvas import FigureCanvas, Panel, grid
from .figure_export import export_canvas
from .nmi_svg_figures import extended_figures
from .plot_panels import grouped_bar_ci, horizontal_interval_plot, matrix_plot, reliability_plot
from .source_data import write_source_data
from .visual_identity import (
    CONSTRUCT_COLORS,
    DATASET_COLORS,
    MODEL_COLORS,
    POLICY_COLORS,
    STATE_COLORS,
    VI,
    dataset_label,
    human_label,
    lighten,
    model_label,
    policy_label,
)


MAIN_W = 2300
MAIN_H = 1500


def _read_csv(name: str) -> pd.DataFrame:
    path = REAL_DATA_EXPERIMENTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_summary() -> dict:
    path = REAL_DATA_EXPERIMENTS_DIR / "real_experiment_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(canvas: FigureCanvas, stem: str) -> list[Path]:
    return export_canvas(canvas, stem)


def _header(canvas: FigureCanvas, title: str, subtitle: str) -> None:
    canvas.text(52, 58, title, size=35, weight="800", fill=VI.ink)
    canvas.text(54, 96, subtitle, size=16.5, fill=VI.muted, max_width=1970)


def _dataset_order() -> list[str]:
    return ["haiid", "chi2023_dke", "convxai_iui2025", "pardos_chatgpt_tutoring", "flora_ips"]


def _metric_label(value: str) -> str:
    labels = {
        "final minus initial accuracy": "Final minus initial accuracy",
        "post-test minus pre-test score": "Post-test minus pre-test score",
        "high versus low metacognitive prompt ratio score": "High vs low metacognitive prompt ratio",
        "overreliance": "Overreliance",
        "underreliance": "Underreliance",
        "correct_ai_reliance": "Correct AI reliance",
        "correct_self_reliance": "Correct self-reliance",
        "appropriate_reliance": "Appropriate reliance",
        "final_correct": "Final correct",
    }
    return labels.get(str(value), human_label(value))


def _clean_condition(value: object) -> str:
    text = human_label(value)
    fixes = {
        "Tutorial XAI": "Tutorial + XAI",
        "Tutorial Only": "Tutorial only",
        "XAI Only": "XAI only",
        "Conversational XAI Boost": "Conversational XAI + boost",
        "LLM Agent Conversational XAI": "LLM-agent conversational XAI",
        "XAI Dashboard": "XAI dashboard",
        "No Hint Control": "No-hint control",
    }
    return fixes.get(text, text)


def _state_label(value: object) -> str:
    labels = {
        "beneficial_ai_reliance": "Beneficial AI reliance",
        "correct_self_reliance": "Correct self-reliance",
        "harmful_ai_overreliance": "Harmful overreliance",
        "harmful_ai_underreliance": "Harmful underreliance",
        "independent_correct": "Independent correct",
        "independent_incorrect": "Independent incorrect",
        "uncertain_disagreement": "Uncertain disagreement",
        "chatgpt_help_learning": "ChatGPT help learning",
        "human_tutor_learning": "Human tutor learning",
        "no_help_learning": "No-help learning",
        "observational_process_trace": "Observational process trace",
        "high_score_metacognitive_process": "High-score metacognitive process",
        "low_score_paste_heavy_process": "Low-score paste-heavy process",
    }
    return labels.get(str(value), human_label(value))


def _rule_label(value: object) -> str:
    labels = {
        "wrong_advice_overreliance": "Wrong-advice overreliance",
        "correct_advice_underreliance": "Correct-advice underreliance",
        "confidence_inflated_reliance": "Confidence-inflated reliance",
        "confident_wrong_advice_vulnerability": "Confident wrong-advice vulnerability",
        "low_confidence_beneficial_reliance": "Low-confidence beneficial reliance",
        "tutorial_protective": "Tutorial protective",
        "process_engagement_protective": "Process engagement protective",
        "advice_source_susceptibility": "Advice-source susceptibility",
        "task_domain_susceptibility": "Task-domain susceptibility",
    }
    return labels.get(str(value), human_label(value))


def _short_dataset_label(value: object) -> str:
    label = dataset_label(value)
    return {"Pardos/Bhandari": "Pardos", "CHI 2023 DKE": "CHI DKE"}.get(label, label)


def _evidence_badge(canvas: FigureCanvas, x: float, y: float, label: str, color: str, detail: str, *, w: float = 275) -> None:
    canvas.round_rect(x, y, w, 106, fill=lighten(color, 0.9), stroke=color, radius=20, width=1.3)
    canvas.dot(x + 25, y + 29, 9, fill=color)
    canvas.text(x + 44, y + 34, label, size=16, weight="800", max_width=w - 56)
    canvas.text(x + 20, y + 67, detail, size=11.5, fill=VI.muted, max_width=w - 34)


def _callout(canvas: FigureCanvas, x: float, y: float, title: str, body: str, color: str, *, w: float = 300, h: float = 118) -> None:
    canvas.round_rect(x, y, w, h, fill=VI.white, stroke=lighten(color, 0.28), radius=18, width=1.5)
    canvas.round_rect(x, y, 8, h, fill=color, radius=4)
    canvas.text(x + 22, y + 32, title, size=15.2, weight="800", max_width=w - 38)
    canvas.text(x + 22, y + 65, body, size=11.5, fill=VI.muted, max_width=w - 38, line_height=1.22)


def _arrow_between(canvas: FigureCanvas, left: Panel, right: Panel, *, color: str = "#8B9BA6") -> None:
    canvas.arrow(left.x + left.w + 7, left.y + left.h / 2, right.x - 7, right.y + right.h / 2, stroke=color, width=2.0)


def _format_ci(row: pd.Series, estimate: str = "effect", low: str = "ci_low", high: str = "ci_high") -> str:
    return f"{float(row[estimate]):.3f} [{float(row[low]):.3f}, {float(row[high]):.3f}]"


def _or_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    for col in ["estimate", "ci_low", "ci_high"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["odds_ratio"] = np.exp(out["estimate"].clip(-6, 6))
    out["or_low"] = np.exp(out["ci_low"].clip(-6, 6))
    out["or_high"] = np.exp(out["ci_high"].clip(-6, 6))
    return out


def _draw_state_stack(canvas: FigureCanvas, p: Panel, state: pd.DataFrame, *, datasets: list[str]) -> None:
    if state.empty:
        canvas.text(p.x + p.w / 2, p.y + p.h / 2, "No state data", size=16, fill=VI.muted, anchor="middle")
        return
    frame = state.loc[state["dataset_name"].isin(datasets)].copy()
    frame["state"] = frame["reliance_state"].map(_state_label)
    frame["share"] = frame["count"] / frame.groupby("dataset_name")["count"].transform("sum")
    write_source_data("figure_02_asymmetric_reliance_failures", "E_state_decomposition", frame)
    states = [
        "Beneficial AI reliance",
        "Correct self-reliance",
        "Independent correct",
        "Independent incorrect",
        "Harmful overreliance",
        "Harmful underreliance",
        "Uncertain disagreement",
    ]
    gap = 28
    legend_w = 310
    bar_w = min(170, max(108, (p.w - legend_w - 90 - gap * (len(datasets) - 1)) / max(len(datasets), 1)))
    left = p.x + 38
    top = p.y + 72
    bar_h = p.h - 155
    for i, dataset in enumerate(datasets):
        x = left + i * (bar_w + gap)
        current = top + bar_h
        sub = frame.loc[frame["dataset_name"].eq(dataset)]
        canvas.text(x + bar_w / 2, p.y + 42, _short_dataset_label(dataset), size=13.5, weight="800", anchor="middle")
        for state_name in states:
            row = sub.loc[sub["state"].eq(state_name)]
            if row.empty:
                continue
            share = float(row["share"].iloc[0])
            h = max(1.2, bar_h * share)
            current -= h
            color = STATE_COLORS.get(state_name, "#868E96")
            canvas.round_rect(x, current, bar_w, h, fill=color, stroke="white", radius=3, width=0.9)
            if h > 28:
                canvas.text(x + bar_w / 2, current + h / 2 + 5, f"{share:.2f}", size=10.5, fill="white", weight="800", anchor="middle")
    legend_x = left + len(datasets) * (bar_w + gap) + 18
    for j, state_name in enumerate(states):
        y = top + j * 31
        color = STATE_COLORS.get(state_name, "#868E96")
        canvas.round_rect(legend_x, y, 22, 18, fill=color, stroke="none", radius=4)
        canvas.text(legend_x + 32, y + 14, state_name, size=10.7, max_width=255)


def figure_1(summary: dict, construct: pd.DataFrame) -> list[Path]:
    stem = "figure_01_evidence_to_action_map"
    canvas = FigureCanvas(MAIN_W, 1450)
    _header(
        canvas,
        "From public evidence to reliance-state diagnosis and selective action",
        "Five public datasets support a bounded framework for measuring decision reliance, short-term learning gain and GenAI process traces without claiming delayed recall, transfer or clinical diagnosis.",
    )
    sizes = summary.get("dataset_sizes", {})
    dataset_rows = []
    for key in _dataset_order():
        counts = sizes.get(key, {})
        dataset_rows.append(
            {
                "dataset": dataset_label(key),
                "records": int(counts.get("n_interactions", 0)),
                "participants": int(counts.get("n_participants", 0)),
            }
        )
    write_source_data(stem, "A_dataset_cards", pd.DataFrame(dataset_rows))

    pA = canvas.panel("A", 54, 140, 1030, 390, "Five-dataset empirical base")
    for idx, row in enumerate(dataset_rows):
        x = pA.x + 32 + (idx % 3) * 315
        y = pA.y + 68 + (idx // 3) * 145
        color = DATASET_COLORS[row["dataset"]]
        _evidence_badge(canvas, x, y, row["dataset"], color, f"{row['records']:,} records; {row['participants']:,} people")
    canvas.text(pA.x + 674, pA.y + 274, "Evidence scope is intentionally heterogeneous: decision advice, tutoring outcomes and process traces are analysed separately.", size=12.3, fill=VI.muted, max_width=310)

    pB = canvas.panel("B", 1130, 140, 1115, 390, "Construct ledger")
    constructs = [
        ("Appropriate reliance", "Decision datasets"),
        ("Overreliance", "Decision datasets"),
        ("Underreliance", "Decision datasets"),
        ("Confidence / calibration", "Decision + interface"),
        ("Short-term learning gain", "Pardos/Bhandari"),
        ("Process traces", "FLoRA IPS"),
        ("Unsupported", "Delayed recall, transfer, diagnosis"),
    ]
    write_source_data(stem, "B_construct_ledger", pd.DataFrame(constructs, columns=["construct", "support"]))
    for idx, (name, support) in enumerate(constructs):
        color = CONSTRUCT_COLORS[name]
        x = pB.x + 34 + (idx % 4) * 260
        y = pB.y + 74 + (idx // 4) * 112
        canvas.round_rect(x, y, 222, 70, fill=lighten(color, 0.86), stroke=color, radius=16, width=1.1)
        canvas.text(x + 16, y + 25, name, size=12.5, weight="800", max_width=194)
        canvas.text(x + 16, y + 51, support, size=10.5, fill=VI.muted, max_width=194)
    if not construct.empty:
        canvas.text(pB.x + 34, pB.y + 318, "The same construct matrix drives figures, tables and claims audit.", size=12.2, fill=VI.muted, max_width=640)

    pC = canvas.panel("C", 54, 585, 950, 430, "Reliance-state model")
    nodes = [
        ("Harmonized inputs", "initial judgement, advice, final judgement, outcome, context", "#0B7285"),
        ("Calibrated estimator", "tabular risk with uncertainty", "#5C7CFA"),
        ("Fuzzy symbolic rules", "state-specific evidence and explanations", "#E67700"),
        ("State probabilities", "beneficial, harmful and uncertain reliance", "#7048E8"),
    ]
    write_source_data(stem, "C_model_layers", pd.DataFrame(nodes, columns=["layer", "description", "color"]))
    node_panels = grid(Panel(pC.x + 42, pC.y + 78, pC.w - 84, 228), 1, 4, gap_x=24)
    for panel, (title, body, color) in zip(node_panels, nodes, strict=False):
        canvas.card(panel, title, body, color=color)
    for left, right in zip(node_panels[:-1], node_panels[1:], strict=False):
        _arrow_between(canvas, left, right)
    canvas.round_rect(pC.x + 96, pC.y + 340, 760, 50, fill=lighten("#7048E8", 0.9), stroke="#7048E8", radius=16)
    canvas.text(pC.x + 476, pC.y + 372, "Fusion: calibrated risk + rule evidence + uncertainty penalty", size=15.5, weight="800", anchor="middle", fill=VI.ink)

    pD = canvas.panel("D", 1040, 585, 620, 430, "Operational outputs")
    outputs = [
        ("Calibrated risk", "warning reliability"),
        ("Rule trace", "why this case is risky"),
        ("Counterfactual", "what would change the state"),
        ("Selective gating", "when to verify, delay or withhold"),
    ]
    write_source_data(stem, "D_outputs", pd.DataFrame(outputs, columns=["output", "purpose"]))
    for idx, (title, body) in enumerate(outputs):
        x = pD.x + 40 + (idx % 2) * 270
        y = pD.y + 80 + (idx // 2) * 135
        _callout(canvas, x, y, title, body, "#0B7285", w=236, h=96)

    pE = canvas.panel("E", 1700, 585, 545, 430, "Evidence boundary")
    boundary = [
        ("Measured", "reliance states, confidence shifts, short-term learning gain"),
        ("Predicted", "risk under strict splits and transfer tests"),
        ("Explained", "rule activations and counterfactual cards"),
        ("Simulated", "observational gating utility, not causal effect"),
    ]
    write_source_data(stem, "E_evidence_boundary", pd.DataFrame(boundary, columns=["status", "meaning"]))
    for idx, (title, body) in enumerate(boundary):
        color = ["#0B7285", "#5C7CFA", "#E67700", "#7048E8"][idx]
        _callout(canvas, pE.x + 36, pE.y + 72 + idx * 79, title, body, color, w=465, h=60)

    pF = canvas.panel("F", 54, 1070, 2190, 275, "Central thesis")
    thesis = (
        "Average AI-assisted gains can hide asymmetric failures: users may over-rely on wrong advice, "
        "under-use correct advice, or improve performance while remaining poorly calibrated. "
        "Reliance-state modeling makes these failures measurable, auditable and actionable for prospective intervention design."
    )
    canvas.text(pF.x + 44, pF.y + 80, thesis, size=21, fill=VI.ink, weight="700", max_width=1770, line_height=1.28)
    _small = [
        ("No clinical diagnosis", "#868E96"),
        ("No delayed recall claim", "#868E96"),
        ("No causal policy claim", "#868E96"),
    ]
    for i, (label, color) in enumerate(_small):
        canvas.pill(pF.x + 54 + i * 250, pF.y + 205, label, fill=lighten(color, 0.86), stroke=color, text_fill=VI.ink, w=205)
    return _save(canvas, stem)


def figure_2(effect: pd.DataFrame, summary: pd.DataFrame, haiid: pd.DataFrame, state: pd.DataFrame, flora: pd.DataFrame) -> list[Path]:
    stem = "figure_02_asymmetric_reliance_failures"
    canvas = FigureCanvas(MAIN_W, 1550)
    _header(
        canvas,
        "Average gains hide asymmetric reliance failures",
        "Outcome improvements coexist with distinct reliance states; FLoRA is kept separate as an observational process-trace boundary.",
    )
    pA = canvas.panel("A", 54, 142, 670, 430, "Outcome gains, not reliance states")
    gains = effect.loc[~effect["dataset"].eq("flora_ips")].copy()
    gains["dataset_display"] = gains["dataset"].map(_short_dataset_label)
    write_source_data(stem, "A_outcome_gains", gains)
    grouped_bar_ci(
        canvas,
        pA.inset(14, 24),
        gains,
        label_col="dataset_display",
        value_col="effect",
        low_col="ci_low",
        high_col="ci_high",
        color_col="dataset_display",
        colors={_short_dataset_label(k): DATASET_COLORS[dataset_label(k)] for k in _dataset_order()},
        ylabel="gain",
        ylim=(-0.03, max(0.22, float(gains["ci_high"].max()) + 0.03 if not gains.empty else 0.22)),
    )

    pB = canvas.panel("B", 760, 142, 690, 430, "FLoRA observational process contrast")
    flora_plot = flora.copy()
    if not flora_plot.empty:
        flora_plot = flora_plot.sort_values("score_difference_high_minus_low", ascending=True).tail(6)
        flora_plot["feature"] = flora_plot["process_feature"].map(human_label)
        write_source_data(stem, "B_flora_process_contrasts", flora_plot)
    horizontal_interval_plot(
        canvas,
        Panel(pB.x + 6, pB.y + 52, pB.w - 12, pB.h - 132),
        flora_plot,
        label_col="feature",
        estimate_col="score_difference_high_minus_low",
        low_col="ci_low",
        high_col="ci_high",
        color=DATASET_COLORS["FLoRA IPS"],
        xlabel="score difference, high minus low",
        xlim=(-0.05, 0.08),
        value_fmt="{:.3f}",
    )
    canvas.text(pB.x + 32, pB.y + 392, "Observational median-split associations; not pre/post gains.", size=11.7, fill=VI.muted, max_width=560)

    pC = canvas.panel("C", 1488, 142, 760, 430, "Reliance failures are asymmetric")
    rel = summary.loc[summary["dataset_name"].isin(["haiid", "convxai_iui2025"])].copy()
    rows = []
    for _, row in rel.iterrows():
        for metric, label, color in [
            ("overreliance_rate", "Overreliance", CONSTRUCT_COLORS["Overreliance"]),
            ("underreliance_rate", "Underreliance", CONSTRUCT_COLORS["Underreliance"]),
        ]:
            rows.append({"dataset": _short_dataset_label(row["dataset_name"]), "metric": label, "rate": row[metric], "color": color})
    fail = pd.DataFrame(rows)
    write_source_data(stem, "C_failure_decomposition", fail)
    left = pC.x + 80
    top = pC.y + 78
    plot_w = pC.w - 140
    plot_h = 270
    canvas.axis(left, top, plot_w, plot_h, ylabel="rate")
    max_rate = max(0.20, fail["rate"].max() * 1.35 if not fail.empty else 0.2)
    for i, dataset in enumerate(fail["dataset"].unique()):
        sub = fail.loc[fail["dataset"].eq(dataset)]
        for j, (_, row) in enumerate(sub.iterrows()):
            x = left + (i + 0.5) * plot_w / 2 + (j - 0.5) * 76
            h = plot_h * float(row["rate"]) / max_rate
            canvas.round_rect(x - 28, top + plot_h - h, 56, h, fill=row["color"], stroke="none", radius=6)
            canvas.text(x, top + plot_h - h - 10, f"{row['rate']:.3f}", size=11, anchor="middle", fill=VI.muted)
        canvas.text(left + (i + 0.5) * plot_w / 2, top + plot_h + 28, dataset, size=12.5, weight="800", anchor="middle")
    canvas.pill(pC.x + 500, pC.y + 90, "Overreliance", fill=lighten(CONSTRUCT_COLORS["Overreliance"], 0.84), stroke=CONSTRUCT_COLORS["Overreliance"], w=145)
    canvas.pill(pC.x + 500, pC.y + 132, "Underreliance", fill=lighten(CONSTRUCT_COLORS["Underreliance"], 0.84), stroke=CONSTRUCT_COLORS["Underreliance"], w=150)

    pD = canvas.panel("D", 54, 630, 800, 430, "HAIID advice-source asymmetry")
    metrics = ["correct_ai_reliance", "correct_self_reliance", "overreliance", "underreliance"]
    advice = haiid.loc[haiid["metric"].isin(metrics)].copy()
    advice = advice.groupby(["advice_source_label", "metric"], as_index=False)[["rate", "ci_low", "ci_high"]].mean()
    advice["source"] = advice["advice_source_label"].map({"ai": "AI", "human": "Human"})
    advice["label"] = advice["metric"].map(_metric_label)
    write_source_data(stem, "D_haiid_advice_source", advice)
    left = pD.x + 235
    top = pD.y + 66
    plot_w = pD.w - 310
    row_h = 72
    canvas.axis(left, top + 10, plot_w, row_h * len(metrics), xlabel="mean rate across task families")
    colors = {"AI": DATASET_COLORS["HAIID"], "Human": "#7C8A96"}
    for i, metric in enumerate(metrics):
        y = top + i * row_h + 32
        canvas.text(pD.x + 28, y + 5, _metric_label(metric), size=12.5, max_width=190)
        for j, source in enumerate(["AI", "Human"]):
            row = advice.loc[(advice["metric"].eq(metric)) & (advice["source"].eq(source))]
            if row.empty:
                continue
            rate = float(row["rate"].iloc[0])
            x = left + plot_w * rate / 0.62
            canvas.line(left, y + j * 16, x, y + j * 16, stroke=colors[source], width=5.5)
            canvas.dot(x, y + j * 16, 6, fill=colors[source])
            canvas.text(x + 10, y + j * 16 + 4, f"{rate:.2f}", size=10.4, fill=VI.muted)
    canvas.pill(pD.x + 603, pD.y + 72, "AI", fill=lighten(colors["AI"], 0.84), stroke=colors["AI"], w=76)
    canvas.pill(pD.x + 686, pD.y + 72, "Human", fill=lighten(colors["Human"], 0.84), stroke=colors["Human"], w=94)

    pE = canvas.panel("E", 895, 630, 835, 430, "Reliance-state decomposition")
    _draw_state_stack(canvas, pE, state, datasets=["haiid", "chi2023_dke", "convxai_iui2025"])

    pF = canvas.panel("F", 1768, 630, 480, 430, "Why accuracy alone is insufficient")
    decomposition = [
        ("Final correct", "can rise when beneficial reliance offsets mistakes", "#2B8A3E"),
        ("Overreliance", "initially correct user moves toward wrong advice", "#C92A2A"),
        ("Underreliance", "initially wrong user fails to use correct advice", "#E67700"),
        ("State mix", "separates helpful, harmful and independent behaviour", "#7048E8"),
    ]
    write_source_data(stem, "F_decomposition_logic", pd.DataFrame(decomposition, columns=["component", "interpretation", "color"]))
    for i, (title, body, color) in enumerate(decomposition):
        _callout(canvas, pF.x + 34, pF.y + 62 + i * 86, title, body, color, w=410, h=72)
    return _save(canvas, stem)


def figure_3(gee: pd.DataFrame, chi_conditions: pd.DataFrame, conv_conditions: pd.DataFrame) -> list[Path]:
    stem = "figure_03_interface_calibration"
    canvas = FigureCanvas(MAIN_W, 1480)
    _header(
        canvas,
        "Interface, advice source and self-calibration shape reliance quality",
        "Clustered GEE estimates and condition contrasts show that trust, confidence and explanation interfaces can be protective or harmful depending on context.",
    )
    gee = gee.copy()
    if not gee.empty:
        gee["display_term"] = gee["term"].map(human_label)
        gee["display_dataset"] = gee["dataset"].map(dataset_label)
        gee["display_outcome"] = gee["outcome"].map(_metric_label)
    gee_or = _or_frame(gee.loc[gee["family"].astype(str).str.lower().eq("binomial")].copy()) if not gee.empty else pd.DataFrame()

    pA = canvas.panel("A", 54, 140, 690, 420, "HAIID advice-source and confidence effects")
    terms = ["C(advice_source_label)[T.human]", "initial_confidence"]
    h = gee_or.loc[(gee_or["dataset"] == "haiid") & (gee_or["term"].isin(terms))].copy()
    h["term_display"] = h["term"].map(
        {
            "C(advice_source_label)[T.human]": "human-labelled advice",
            "initial_confidence": "initial confidence",
        }
    )
    h["label"] = h.apply(lambda r: f"{_metric_label(r['outcome'])}: {r['term_display']}", axis=1)
    write_source_data(stem, "A_haiid_gee", h)
    horizontal_interval_plot(
        canvas,
        pA.inset(0, 24),
        h.sort_values("odds_ratio"),
        label_col="label",
        estimate_col="odds_ratio",
        low_col="or_low",
        high_col="or_high",
        color=DATASET_COLORS["HAIID"],
        xlabel="odds ratio",
        zero=1,
        xlim=(0, 7.2),
        value_fmt="{:.2f}",
    )

    pB = canvas.panel("B", 780, 140, 690, 420, "CHI intervention contrasts")
    chi = chi_conditions.loc[chi_conditions["metric"].eq("appropriate_reliance")].copy()
    chi["condition"] = chi["condition_name"].map(_clean_condition)
    write_source_data(stem, "B_chi_conditions", chi)
    grouped_bar_ci(
        canvas,
        pB.inset(12, 24),
        chi,
        label_col="condition",
        value_col="rate",
        low_col="ci_low",
        high_col="ci_high",
        color_col=None,
        colors=None,
        ylabel="appropriate reliance",
        ylim=(0.0, 0.68),
    )

    pC = canvas.panel("C", 1508, 140, 740, 420, "Self-assessment calibration in CHI")
    chi_terms = ["first_batch_overestimation", "first_batch_underestimation", "trust_first"]
    chig = gee_or.loc[(gee_or["dataset"] == "chi2023_dke") & (gee_or["term"].isin(chi_terms))].copy()
    chig["label"] = chig["term"].map(
        {
            "first_batch_overestimation": "self-overestimation",
            "first_batch_underestimation": "self-underestimation",
            "trust_first": "initial trust",
        }
    )
    write_source_data(stem, "C_chi_gee", chig)
    horizontal_interval_plot(
        canvas,
        pC.inset(8, 24),
        chig.sort_values("odds_ratio"),
        label_col="label",
        estimate_col="odds_ratio",
        low_col="or_low",
        high_col="or_high",
        color=DATASET_COLORS["CHI 2023 DKE"],
        xlabel="odds ratio for appropriate reliance",
        zero=1,
        xlim=(0.25, 2.4),
        value_fmt="{:.2f}",
    )

    pD = canvas.panel("D", 54, 625, 970, 435, "ConvXAI interface contrasts")
    conv = conv_conditions.loc[conv_conditions["metric"].eq("appropriate_reliance")].copy()
    conv["condition"] = conv["condition_name"].map(_clean_condition)
    write_source_data(stem, "D_convxai_conditions", conv)
    horizontal_interval_plot(
        canvas,
        pD.inset(2, 24),
        conv.sort_values("rate"),
        label_col="condition",
        estimate_col="rate",
        low_col="ci_low",
        high_col="ci_high",
        color=DATASET_COLORS["ConvXAI"],
        xlabel="appropriate reliance rate",
        zero=0.5,
        xlim=(0.34, 0.72),
        value_fmt="{:.2f}",
    )

    pE = canvas.panel("E", 1062, 625, 680, 435, "Reliability can cut both ways")
    conv_terms = gee.loc[(gee["dataset"] == "convxai_iui2025") & (gee["term"] == "post_explain_reliability")].copy()
    conv_terms["label"] = conv_terms["outcome"].map(_metric_label)
    write_source_data(stem, "E_reliability_gee", conv_terms)
    horizontal_interval_plot(
        canvas,
        pE.inset(4, 24),
        conv_terms,
        label_col="label",
        estimate_col="estimate",
        low_col="ci_low",
        high_col="ci_high",
        color=CONSTRUCT_COLORS["Confidence / calibration"],
        xlabel="GEE log-odds per reliability signal",
        zero=0,
        xlim=(-1.0, 10.0),
        value_fmt="{:.2f}",
    )
    canvas.text(pE.x + 42, pE.y + 382, "Large overreliance association is shown, not hidden; it motivates selective warnings.", size=11.5, fill=VI.muted, max_width=560)

    pF = canvas.panel("F", 1780, 625, 468, 435, "Interpretation boundary")
    messages = [
        ("Trust", "can support appropriate reliance"),
        ("Confidence", "can protect against wrong advice or block correct advice"),
        ("Reliability", "can improve reliance or amplify overreliance"),
        ("Design", "must be evaluated by state, not only accuracy"),
    ]
    write_source_data(stem, "F_interpretation_boundary", pd.DataFrame(messages, columns=["signal", "interpretation"]))
    for i, (title, body) in enumerate(messages):
        _callout(canvas, pF.x + 34, pF.y + 58 + i * 88, title, body, ["#0B7285", "#E67700", "#7048E8", "#5C7CFA"][i], w=400, h=74)
    return _save(canvas, stem)


def figure_4() -> list[Path]:
    stem = "figure_04_reliance_state_formalism"
    canvas = FigureCanvas(MAIN_W, 1510)
    _header(
        canvas,
        "ReliaGuard-NS conformal neuro-symbolic reliance control",
        "The method separates measurement, prediction, split-conformal selective-risk control and observational policy simulation so diagnostic value is not confused with causal intervention evidence.",
    )

    pA = canvas.panel("A", 54, 140, 610, 390, "State formalism")
    states = [
        ("Beneficial AI reliance", "h0 wrong, advice correct, h1 moves correct"),
        ("Harmful overreliance", "h0 correct, advice wrong, h1 moves wrong"),
        ("Harmful underreliance", "h0 wrong, advice correct, h1 resists"),
        ("Correct self-reliance", "h0 correct, advice wrong, h1 resists"),
        ("Uncertain disagreement", "low-confidence or ambiguous movement"),
    ]
    write_source_data(stem, "A_state_definitions", pd.DataFrame(states, columns=["state", "definition"]))
    for idx, (state_name, definition) in enumerate(states):
        color = STATE_COLORS.get(state_name, "#868E96")
        _callout(canvas, pA.x + 34, pA.y + 62 + idx * 61, state_name, definition, color, w=540, h=49)

    pB = canvas.panel("B", 704, 140, 635, 390, "Fuzzy rule layer")
    formulae = [
        ("Activation", "rₖ(x) = Π μₖⱼ(xⱼ)"),
        ("Rule evidence", "Rₛ(x) = Σ wₖ rₖ(x)"),
        ("Example", "high confidence + wrong advice movement → overreliance risk"),
    ]
    write_source_data(stem, "B_rule_formulae", pd.DataFrame(formulae, columns=["component", "expression"]))
    for i, (title, expression) in enumerate(formulae):
        color = ["#E67700", "#E67700", "#C92A2A"][i]
        canvas.round_rect(pB.x + 48, pB.y + 76 + i * 92, pB.w - 96, 64, fill=lighten(color, 0.9), stroke=color, radius=16)
        canvas.text(pB.x + 72, pB.y + 103 + i * 92, title, size=14, weight="800")
        canvas.text(pB.x + 230, pB.y + 104 + i * 92, expression, size=15.5, fill=VI.ink, max_width=360)

    pC = canvas.panel("C", 1380, 140, 868, 390, "Calibration-aware fusion")
    blocks = [
        ("Tabular estimator", "ηθ(x)", "#5C7CFA"),
        ("Rule evidence", "Rₛ(x)", "#E67700"),
        ("Uncertainty", "u(x)", "#7048E8"),
        ("State risk", "pₛ(x)", "#0B7285"),
    ]
    write_source_data(stem, "C_fusion_blocks", pd.DataFrame(blocks, columns=["block", "symbol", "color"]))
    block_panels = grid(Panel(pC.x + 46, pC.y + 78, pC.w - 92, 120), 1, 4, gap_x=24)
    for panel, (name, symbol, color) in zip(block_panels, blocks, strict=False):
        canvas.card(panel, name, symbol, color=color)
    for left, right in zip(block_panels[:-1], block_panels[1:], strict=False):
        _arrow_between(canvas, left, right, color="#7C8A96")
    canvas.round_rect(pC.x + 82, pC.y + 260, pC.w - 164, 74, fill=lighten("#0B7285", 0.9), stroke="#0B7285", radius=18)
    canvas.text(pC.x + pC.w / 2, pC.y + 304, "pₛ(x) = σ(ηθ(x) + λRₛ(x) − γu(x))", size=22, weight="800", anchor="middle")

    pD = canvas.panel("D", 54, 590, 760, 420, "Accuracy decomposition")
    identity = [
        ("Initial accuracy", "A₀"),
        ("Final accuracy", "A₁ = A₀ + beneficial moves − harmful moves"),
        ("Failure terms", "overreliance and underreliance can remain non-zero"),
        ("Implication", "accuracy gains do not imply appropriate reliance"),
    ]
    write_source_data(stem, "D_accuracy_decomposition", pd.DataFrame(identity, columns=["term", "statement"]))
    canvas.round_rect(pD.x + 48, pD.y + 92, pD.w - 96, 78, fill=lighten("#0B7285", 0.88), stroke="#0B7285", radius=18)
    canvas.text(pD.x + pD.w / 2, pD.y + 140, "ΔA = gains from corrected errors − losses from induced errors", size=20, weight="800", anchor="middle")
    canvas.text(pD.x + 54, pD.y + 230, "Because both positive and negative state transitions contribute to final accuracy, a positive ΔA can coexist with substantial overreliance and underreliance.", size=16.5, fill=VI.ink, max_width=pD.w - 108, line_height=1.28)

    pE = canvas.panel("E", 850, 590, 720, 420, "Gating utility")
    utility = [
        ("Objective", "U(a) = P(correct|a) − αP(over|a) − βP(under|a) − κI(a)"),
        ("Actions", "accept, resist, verify, delay, withhold"),
        ("Conformal control", "threshold τ controls missed harmful cases under exchangeability"),
    ]
    write_source_data(stem, "E_gating_utility", pd.DataFrame(utility, columns=["component", "statement"]))
    for i, (title, statement) in enumerate(utility):
        _callout(canvas, pE.x + 42, pE.y + 80 + i * 96, title, statement, ["#7048E8", "#5C7CFA", "#0B7285"][i], w=632, h=72)

    pF = canvas.panel("F", 1608, 590, 640, 420, "Inference trace")
    trace = [
        ("Inputs", "initial answer, advice, final answer, confidence, context"),
        ("Active rules", "wrong-advice vulnerability + uncertainty"),
        ("Predicted state", "harmful overreliance risk"),
        ("Recommendation", "request evidence comparison before final answer"),
    ]
    write_source_data(stem, "F_inference_trace", pd.DataFrame(trace, columns=["stage", "example"]))
    y_positions = []
    for i, (stage, example) in enumerate(trace):
        y = pF.y + 75 + i * 76
        y_positions.append(y + 28)
        canvas.round_rect(pF.x + 48, y, pF.w - 96, 56, fill=VI.white, stroke="#D6DEE4", radius=14)
        canvas.text(pF.x + 70, y + 24, stage, size=13.5, weight="800")
        canvas.text(pF.x + 225, y + 24, example, size=11.8, fill=VI.muted, max_width=360)
    for y1, y2 in zip(y_positions[:-1], y_positions[1:], strict=False):
        canvas.arrow(pF.x + pF.w / 2, y1 + 8, pF.x + pF.w / 2, y2 - 24, stroke="#9AA8B2", width=1.8)

    pG = canvas.panel("G", 54, 1060, 2194, 270, "Boundary of claim")
    boundaries = [
        ("Measurement", "state labels from observed initial advice final outcome tuples"),
        ("Prediction", "risk under strict participant, task and transfer splits"),
        ("Explanation", "rule trace and counterfactual features"),
        ("Policy simulation", "offline utility analysis and ReliaGuard-NS thresholds, explicitly not causal"),
    ]
    write_source_data(stem, "G_claim_boundary", pd.DataFrame(boundaries, columns=["claim_type", "scope"]))
    for i, (title, body) in enumerate(boundaries):
        _callout(canvas, pG.x + 42 + i * 535, pG.y + 88, title, body, ["#0B7285", "#5C7CFA", "#E67700", "#7048E8"][i], w=485, h=92)
    return _save(canvas, stem)


def figure_5(model: pd.DataFrame, cross: pd.DataFrame, curves: pd.DataFrame, calibration: pd.DataFrame, decision_curve: pd.DataFrame, reliaguard: pd.DataFrame) -> list[Path]:
    stem = "figure_05_prediction_calibration_transfer"
    canvas = FigureCanvas(MAIN_W, 1530)
    _header(
        canvas,
        "Prediction, calibration, conformal control and transfer define what generalizes",
        "Strict splits retain strong baselines, calibration is mixed across settings, and ReliaGuard-NS reports selective-risk trade-offs under explicit exchangeability assumptions.",
    )

    pA = canvas.panel("A", 54, 140, 760, 430, "Strict validation model comparison")
    if not model.empty:
        strict = model.loc[model["split"].isin(["participant", "strict_task"])].copy()
        keep_models = [
            "logistic_regression",
            "calibrated_gradient_boosting",
            "symbolic_only",
            "uncertainty_aware_fusion",
            "reliance_state_neurosymbolic",
        ]
        strict = strict.loc[strict["model"].isin(keep_models)]
        strict = strict.sort_values("auroc", ascending=False).groupby("model", as_index=False).head(1)
        strict["model_display"] = strict["model"].map(model_label)
        strict["label"] = strict["model_display"] + " (" + strict["dataset"].map(_short_dataset_label) + ")"
    else:
        strict = pd.DataFrame()
    write_source_data(stem, "A_strict_model_comparison", strict)
    horizontal_interval_plot(
        canvas,
        pA.inset(4, 24),
        strict.sort_values("auroc") if not strict.empty else strict,
        label_col="label",
        estimate_col="auroc",
        low_col="auroc_ci_low",
        high_col="auroc_ci_high",
        color="#7048E8",
        xlabel="AUROC under strict validation",
        zero=0.5,
        xlim=(0.48, 1.0),
        value_fmt="{:.2f}",
    )

    pB = canvas.panel("B", 850, 140, 640, 430, "Delta against best non-symbolic baseline")
    deltas = []
    if not model.empty:
        grouped = model.loc[model["split"].isin(["participant", "strict_task"])].copy()
        for (dataset, target, split), sub in grouped.groupby(["dataset", "target", "split"]):
            ns = sub.loc[sub["model"].eq("reliance_state_neurosymbolic")]
            base = sub.loc[~sub["model"].isin(["reliance_state_neurosymbolic", "symbolic_only"])].sort_values("auroc", ascending=False)
            if not ns.empty and not base.empty:
                deltas.append(
                    {
                        "setting": f"{_short_dataset_label(dataset)} {human_label(target)}",
                        "delta": float(ns["auroc"].iloc[0] - base["auroc"].iloc[0]),
                    }
                )
    delta_df = pd.DataFrame(deltas).sort_values("delta") if deltas else pd.DataFrame(columns=["setting", "delta"])
    write_source_data(stem, "B_delta_vs_baseline", delta_df)
    if not delta_df.empty:
        left = pB.x + 230
        top = pB.y + 62
        plot_w = pB.w - 285
        row_h = min(44, (pB.h - 120) / len(delta_df))
        canvas.axis(left, top, plot_w, row_h * len(delta_df), xlabel="AUROC delta")
        zero_x = left + plot_w * 0.5
        canvas.line(zero_x, top, zero_x, top + row_h * len(delta_df), stroke=VI.ink, width=1.0, opacity=0.5)
        for i, row in delta_df.iterrows():
            y = top + row_h * (list(delta_df.index).index(i) + 0.5)
            delta = float(row["delta"])
            x = zero_x + delta / 0.16 * plot_w * 0.48
            color = "#2B8A3E" if delta >= 0 else "#C92A2A"
            canvas.text(pB.x + 24, y + 4, row["setting"], size=10.5, max_width=190)
            canvas.line(zero_x, y, x, y, stroke=color, width=4.5)
            canvas.dot(x, y, 6, fill=color)
            canvas.text(x + (8 if delta >= 0 else -8), y + 4, f"{delta:+.2f}", size=10.5, fill=VI.muted, anchor="start" if delta >= 0 else "end")

    pC = canvas.panel("C", 1526, 140, 722, 430, "Cross-dataset transfer")
    if not cross.empty:
        target_cross = cross.loc[cross["target"].eq("appropriate_reliance")].copy()
        target_cross = target_cross.sort_values("auroc", ascending=False).groupby(["train_dataset", "test_dataset"], as_index=False).head(1)
        train = ["haiid", "convxai_iui2025", "chi2023_dke"]
        test = ["haiid", "convxai_iui2025", "chi2023_dke"]
        mat = pd.DataFrame(index=[_short_dataset_label(x) for x in train], columns=[_short_dataset_label(x) for x in test], dtype=float)
        for _, row in target_cross.iterrows():
            if row["train_dataset"] in train and row["test_dataset"] in test:
                mat.loc[_short_dataset_label(row["train_dataset"]), _short_dataset_label(row["test_dataset"])] = row["auroc"]
    else:
        target_cross = pd.DataFrame()
        mat = pd.DataFrame()
    write_source_data(stem, "C_cross_dataset_transfer", target_cross)
    matrix_plot(canvas, pC.inset(12, 20), mat, row_labels=list(mat.index), col_labels=list(mat.columns), lo=0.45, hi=0.82)

    pD = canvas.panel("D", 54, 635, 730, 430, "Reliability curves: strong and difficult cases")
    curve_rows = pd.DataFrame()
    if not curves.empty:
        strong = curves.loc[
            (curves["dataset"] == "haiid")
            & (curves["target"] == "underreliance")
            & (curves["split"] == "random")
            & (curves["model"].isin(["uncertainty_aware_fusion", "logistic_regression"]))
        ].copy()
        hard = curves.loc[
            (curves["dataset"] == "convxai_iui2025")
            & (curves["target"] == "overreliance")
            & (curves["split"] == "participant")
            & (curves["model"].isin(["reliance_state_neurosymbolic", "calibrated_gradient_boosting"]))
        ].copy()
        curve_rows = pd.concat([strong.assign(panel="strong"), hard.assign(panel="difficult")], ignore_index=True)
        curve_rows["model_display"] = curve_rows["model"].map(model_label)
    write_source_data(stem, "D_reliability_curves", curve_rows)
    subpanels = grid(pD.inset(6, 24), 1, 2, gap_x=26)
    for panel, label in zip(subpanels, ["strong", "difficult"], strict=False):
        sub = curve_rows.loc[curve_rows["panel"].eq(label)].copy()
        title = "Lower ECE case" if label == "strong" else "Harder calibration case"
        reliability_plot(
            canvas,
            panel,
            sub,
            model_col="model_display",
            x_col="mean_confidence",
            y_col="accuracy",
            models=list(sub["model_display"].dropna().unique())[:2],
            colors={model_label(k): MODEL_COLORS.get(model_label(k), "#7048E8") for k in sub["model"].dropna().unique()},
            title=title,
        )

    pE = canvas.panel("E", 820, 635, 685, 430, "Calibration summary")
    cal = calibration.copy()
    if not cal.empty:
        cal["model_display"] = cal["model"].map(model_label)
        cal = cal.sort_values("ece").groupby(["dataset", "target", "split"], as_index=False).head(1).head(10)
        cal["label"] = cal["dataset"].map(_short_dataset_label) + ": " + cal["target"].map(_metric_label)
    write_source_data(stem, "E_calibration_summary", cal)
    horizontal_interval_plot(
        canvas,
        pE.inset(0, 24),
        cal.sort_values("ece", ascending=False) if not cal.empty else cal,
        label_col="label",
        estimate_col="ece",
        low_col="ece",
        high_col="ece",
        color="#0B7285",
        xlabel="expected calibration error",
        zero=0,
        xlim=(0, 0.28),
        value_fmt="{:.2f}",
    )

    pF = canvas.panel("F", 1540, 635, 708, 430, "ReliaGuard-NS selective-risk frontier")
    rg = reliaguard.copy()
    if not rg.empty:
        rg = rg.loc[rg["model"].eq("reliance_state_neurosymbolic")].copy()
        if rg.empty:
            rg = reliaguard.copy()
        rg = rg.sort_values(["dataset", "target", "intervention_burden"]).groupby(["dataset", "target"]).head(1)
        rg["target_display"] = rg["target"].map(_metric_label)
    write_source_data(stem, "F_reliaguard_selective_risk", rg)
    if not rg.empty:
        left = pF.x + 72
        top = pF.y + 60
        plot_w = pF.w - 130
        plot_h = pF.h - 120
        canvas.axis(left, top, plot_w, plot_h, xlabel="intervention burden", ylabel="harm among non-intervened")
        burden_max = max(0.62, float(rg["intervention_burden"].max()) + 0.10)
        harm_max = max(0.25, float(rg["harmful_rate_among_non_intervened"].max()) + 0.05)
        for dataset in ["haiid", "chi2023_dke", "convxai_iui2025"]:
            sub = rg.loc[rg["dataset"].eq(dataset)].sort_values("intervention_burden")
            color = DATASET_COLORS[dataset_label(dataset)]
            for _, row in sub.iterrows():
                x = left + plot_w * float(row["intervention_burden"]) / burden_max
                y = top + plot_h - plot_h * float(row["harmful_rate_among_non_intervened"]) / harm_max
                canvas.dot(x, y, 8, fill=color)
                canvas.text(x + 9, y + 3, f"{_short_dataset_label(dataset)} {row['target_display']}", size=9.5, fill=color, max_width=150)
        canvas.text(pF.x + 60, pF.y + 382, "Thresholds control missed harmful cases under exchangeability; they do not identify intervention effects.", size=11.5, fill=VI.muted, max_width=560)
    return _save(canvas, stem)


def figure_6(ablation: pd.DataFrame, policy: pd.DataFrame, decision_curve: pd.DataFrame) -> list[Path]:
    stem = "figure_06_rules_to_policy_frontier"
    canvas = FigureCanvas(MAIN_W, 1540)
    _header(
        canvas,
        "Symbolic rules convert reliance predictions into auditable intervention candidates",
        "Rule groups expose failure modes, counterfactuals translate model states, and policy frontiers quantify observational utility trade-offs.",
    )

    pA = canvas.panel("A", 54, 140, 780, 430, "Rule ablation signal")
    abl = ablation.copy()
    if not abl.empty:
        abl["rule"] = abl["group"].map(_rule_label)
        abl["dataset_target"] = abl["dataset"].map(_short_dataset_label) + ": " + abl["target"].map(_metric_label)
        abl["label"] = abl["rule"] + " (" + abl["dataset"].map(_short_dataset_label) + ")"
        abl = abl.sort_values("mean_auroc_delta").head(9)
    write_source_data(stem, "A_rule_ablation", abl)
    horizontal_interval_plot(
        canvas,
        pA.inset(0, 24),
        abl,
        label_col="label",
        estimate_col="mean_auroc_delta",
        low_col="ci_low",
        high_col="ci_high",
        color="#E67700",
        xlabel="AUROC change after removing rule group",
        zero=0,
        xlim=(-0.34, 0.08),
        value_fmt="{:.2f}",
    )

    pB = canvas.panel("B", 872, 140, 640, 430, "Rule to failure-mode flow")
    flows = [
        ("Wrong-advice rule", "Overreliance", "Request verification", "#C92A2A"),
        ("Correct-advice rule", "Underreliance", "Compare evidence", "#E67700"),
        ("Confidence inflation", "Overreliance", "Show uncertainty cue", "#7048E8"),
        ("Low confidence", "Beneficial reliance", "Support acceptance", "#0B7285"),
    ]
    write_source_data(stem, "B_rule_flow", pd.DataFrame(flows, columns=["rule_group", "failure_mode", "intervention", "color"]))
    xs = [pB.x + 72, pB.x + 300, pB.x + 508]
    headers = ["Rule group", "State", "Action"]
    for x, header in zip(xs, headers, strict=False):
        canvas.text(x, pB.y + 62, header, size=13, weight="800", anchor="middle")
    for i, (rule, state_name, action, color) in enumerate(flows):
        y = pB.y + 108 + i * 70
        canvas.round_rect(xs[0] - 80, y, 160, 38, fill=lighten(color, 0.88), stroke=color, radius=13)
        canvas.text(xs[0], y + 24, rule, size=10.7, weight="700", anchor="middle", max_width=142)
        canvas.round_rect(xs[1] - 78, y, 156, 38, fill=VI.white, stroke=color, radius=13)
        canvas.text(xs[1], y + 24, state_name, size=10.8, weight="700", anchor="middle", max_width=140)
        canvas.round_rect(xs[2] - 86, y, 172, 38, fill=lighten("#0B7285", 0.9), stroke="#0B7285", radius=13)
        canvas.text(xs[2], y + 24, action, size=10.6, weight="700", anchor="middle", max_width=154)
        canvas.arrow(xs[0] + 84, y + 19, xs[1] - 84, y + 19, stroke=color, width=1.7)
        canvas.arrow(xs[1] + 82, y + 19, xs[2] - 92, y + 19, stroke=color, width=1.7)

    pC = canvas.panel("C", 1548, 140, 700, 430, "Counterfactual explanation card")
    card_x, card_y = pC.x + 42, pC.y + 72
    canvas.round_rect(card_x, card_y, pC.w - 84, 280, fill=VI.white, stroke="#D6DEE4", radius=24, width=1.3)
    canvas.text(card_x + 28, card_y + 42, "Case: high-confidence disagreement with low verification", size=17, weight="800", max_width=560)
    explanation = [
        ("Predicted state", "harmful overreliance risk"),
        ("Activated rules", "wrong-advice vulnerability; confidence inflation"),
        ("Counterfactual", "risk decreases if evidence comparison is completed before final answer"),
        ("Action", "request verification rather than suppress all advice"),
    ]
    write_source_data(stem, "C_counterfactual_card", pd.DataFrame(explanation, columns=["field", "text"]))
    for i, (field, text) in enumerate(explanation):
        y = card_y + 82 + i * 45
        canvas.text(card_x + 30, y, field, size=12.2, weight="800", fill=VI.ink)
        canvas.text(card_x + 178, y, text, size=12.2, fill=VI.muted, max_width=420)
    canvas.text(pC.x + 50, pC.y + 383, "Explanation is diagnostic and prospective; it is not a demonstrated causal effect.", size=11.8, fill=VI.muted, max_width=600)

    pD = canvas.panel("D", 54, 635, 700, 430, "Intervention distribution")
    pol = policy.copy()
    if not pol.empty:
        pol["policy_display"] = pol["policy"].map(policy_label)
        pol = pol.loc[pol["policy"].eq("neurosymbolic_gating")]
        pol["action"] = pol["dataset"].map(_short_dataset_label)
    write_source_data(stem, "D_policy_distribution", pol)
    if not pol.empty:
        left = pD.x + 90
        top = pD.y + 68
        plot_w = pD.w - 165
        plot_h = pD.h - 125
        canvas.axis(left, top, plot_w, plot_h, ylabel="burden")
        burden_max = max(1.05, float(pol["intervention_burden"].max()) + 0.06)
        for i, (_, row) in enumerate(pol.iterrows()):
            x = left + (i + 0.5) * plot_w / len(pol)
            burden = float(row["intervention_burden"])
            h = min(plot_h, plot_h * burden / burden_max)
            color = DATASET_COLORS[dataset_label(row["dataset"])]
            canvas.round_rect(x - 38, top + plot_h - h, 76, h, fill=color, stroke="none", radius=7)
            canvas.text(x, top + plot_h - h - 10, f"{burden:.2f}", size=11, anchor="middle", fill=VI.muted)
            canvas.text(x, top + plot_h + 28, _short_dataset_label(row["dataset"]), size=11.5, weight="800", anchor="middle")

    pE = canvas.panel("E", 792, 635, 780, 430, "Offline policy frontier")
    dc = decision_curve.copy()
    if not dc.empty:
        dc["policy_display"] = dc["policy"].map(policy_label)
    write_source_data(stem, "E_policy_frontier", dc)
    if not dc.empty:
        left = pE.x + 78
        top = pE.y + 60
        plot_w = pE.w - 145
        plot_h = pE.h - 118
        canvas.axis(left, top, plot_w, plot_h, xlabel="intervention burden", ylabel="utility")
        burden_max = max(0.62, float(dc["intervention_burden"].max()) + 0.10)
        utility_min = min(-0.16, float(dc["expected_utility"].min()) - 0.02)
        utility_max = max(0.10, float(dc["expected_utility"].max()) + 0.03)
        utility_span = utility_max - utility_min or 0.1
        for dataset in ["haiid", "chi2023_dke", "convxai_iui2025"]:
            sub = dc.loc[dc["dataset"].eq(dataset)].sort_values("intervention_burden")
            color = DATASET_COLORS[dataset_label(dataset)]
            pts = []
            for _, row in sub.iterrows():
                x = left + plot_w * float(row["intervention_burden"]) / burden_max
                y = top + plot_h - plot_h * (float(row["expected_utility"]) - utility_min) / utility_span
                pts.append((x, y, row["policy_display"]))
                canvas.dot(x, y, 6, fill=color)
            if len(pts) > 1:
                canvas.dwg.add(canvas.dwg.polyline(points=[(x, y) for x, y, _ in pts], fill="none", stroke=color, stroke_width=2.4))
            if pts:
                canvas.text(pts[-1][0] + 8, pts[-1][1] + 4, _short_dataset_label(dataset), size=10.7, fill=color)
        canvas.text(pE.x + 460, pE.y + 382, "Observational simulation, not causal deployment evidence.", size=11.5, fill=VI.muted, max_width=260)

    pF = canvas.panel("F", 1610, 635, 638, 430, "Utility sensitivity")
    if not decision_curve.empty:
        best = decision_curve.loc[decision_curve["policy"].eq("neurosymbolic_gating")].copy()
        best["utility_gain"] = best["utility_gain_vs_observed"]
        best["label"] = best["dataset"].map(_short_dataset_label)
    else:
        best = pd.DataFrame()
    write_source_data(stem, "F_utility_gain", best)
    horizontal_interval_plot(
        canvas,
        pF.inset(0, 24),
        best.sort_values("utility_gain") if not best.empty else best,
        label_col="label",
        estimate_col="utility_gain",
        low_col="utility_gain",
        high_col="utility_gain",
        color="#7048E8",
        xlabel="utility gain versus observed behaviour",
        zero=0,
        xlim=(-0.03, 0.06),
        value_fmt="{:.3f}",
    )
    return _save(canvas, stem)


def generate_flagship_figures() -> list[Path]:
    ensure_directories()
    summary_json = _read_summary()
    summary = _read_csv("cross_dataset_summary.csv")
    construct = _read_csv("construct_matrix.csv")
    effect = _read_csv("effect_sizes.csv")
    haiid = _read_csv("haiid_descriptive_metrics.csv")
    chi_conditions = _read_csv("chi2023_condition_effects.csv")
    convxai_conditions = _read_csv("convxai_condition_effects.csv")
    gee = _read_csv("gee_results.csv")
    flora = _read_csv("flora_process_effects.csv")
    model = _read_csv("real_model_results.csv")
    cross = _read_csv("cross_dataset_results.csv")
    curves = _read_csv("real_calibration_curves.csv")
    calibration = _read_csv("calibration_summary.csv")
    ablation = _read_csv("ablation_summary.csv")
    policy = _read_csv("policy_evaluation.csv")
    decision_curve = _read_csv("decision_curve_summary.csv")
    reliaguard = _read_csv("reliaguard_conformal_results.csv")
    state = _read_csv("reliance_state_distribution.csv")
    sensitivity = _read_csv("sensitivity_summary.csv")

    outputs: list[Path] = []
    outputs.extend(figure_1(summary_json, construct))
    outputs.extend(figure_2(effect, summary, haiid, state, flora))
    outputs.extend(figure_3(gee, chi_conditions, convxai_conditions))
    outputs.extend(figure_4())
    outputs.extend(figure_5(model, cross, curves, calibration, decision_curve, reliaguard))
    outputs.extend(figure_6(ablation, policy, decision_curve))
    outputs.extend(extended_figures(summary_json, state, model, calibration, cross, sensitivity))
    return outputs
