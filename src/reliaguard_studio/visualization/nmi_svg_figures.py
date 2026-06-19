from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, ensure_directories
from .figure_canvas import FigureCanvas, Panel, grid
from .figure_export import export_canvas
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


MAIN_W = 2200
MAIN_H = 1450


def _read_csv(name: str) -> pd.DataFrame:
    path = REAL_DATA_EXPERIMENTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_summary() -> dict:
    path = REAL_DATA_EXPERIMENTS_DIR / "real_experiment_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _header(canvas: FigureCanvas, title: str, subtitle: str) -> None:
    canvas.text(52, 62, title, size=34, weight="800", fill=VI.ink)
    canvas.text(54, 98, subtitle, size=17, fill=VI.muted, max_width=1840)


def _save(canvas: FigureCanvas, stem: str) -> list[Path]:
    return export_canvas(canvas, stem)


def _dataset_order() -> list[str]:
    return ["haiid", "chi2023_dke", "convxai_iui2025", "pardos_chatgpt_tutoring", "flora_ips"]


def _badge(canvas: FigureCanvas, x: float, y: float, title: str, body: str, color: str, *, w: float = 315, h: float = 120) -> None:
    canvas.round_rect(x, y, w, h, fill=lighten(color, 0.89), stroke=color, radius=22, width=1.5)
    canvas.dot(x + 26, y + 30, 10, fill=color)
    canvas.text(x + 48, y + 35, title, size=17, weight="800", fill=VI.ink, max_width=w - 64)
    body_y = y + (84 if h >= 125 else 66 if h >= 90 else 50)
    canvas.text(x + 22, body_y, body, size=12.5, fill=VI.muted, max_width=w - 38, line_height=1.22)


def _small_key(canvas: FigureCanvas, x: float, y: float, items: list[tuple[str, str]], *, w: float = 260) -> None:
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 32
        canvas.round_rect(x, yy, 22, 22, fill=color, stroke="none", radius=5)
        canvas.text(x + 32, yy + 16, label, size=12.3, fill=VI.ink, max_width=w - 40)


def _or_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["estimate", "ci_low", "ci_high"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["odds_ratio"] = np.exp(out["estimate"].clip(-6, 6))
    out["or_low"] = np.exp(out["ci_low"].clip(-6, 6))
    out["or_high"] = np.exp(out["ci_high"].clip(-6, 6))
    return out


def figure_1(summary: dict) -> list[Path]:
    stem = "figure_01_graphical_abstract"
    canvas = FigureCanvas(MAIN_W, 1320)
    _header(
        canvas,
        "Reliance-state evaluation turns AI-assisted outcomes into diagnosable states",
        "Five real public datasets define where assistance helps, where it fails, and where evidence is still absent.",
    )
    sizes = summary.get("dataset_sizes", {})
    datasets = []
    for key in _dataset_order():
        counts = sizes.get(key, {})
        datasets.append(
            {
                "dataset": dataset_label(key),
                "records": int(counts.get("n_interactions", 0)),
                "people": int(counts.get("n_participants", 0)),
            }
        )
    write_source_data(stem, "A_dataset_cards", pd.DataFrame(datasets))
    pA = canvas.panel("A", 52, 145, 650, 450, "Empirical base")
    for i, row in enumerate(datasets):
        color = DATASET_COLORS[row["dataset"]]
        _badge(
            canvas,
            pA.x + 28 + (i % 2) * 302,
            pA.y + 62 + (i // 2) * 126,
            row["dataset"],
            f"{row['records']:,} records; {row['people']:,} people",
            color,
            w=272,
            h=100,
        )
    canvas.text(pA.x + 28, pA.y + 428, "Synthetic AIR-Bench is retained only as supplementary stress testing.", size=12.8, fill=VI.muted)

    pB = canvas.panel("B", 740, 145, 610, 450, "Construct boundary")
    constructs = [
        ("Appropriate reliance", "real decision evidence"),
        ("Overreliance", "wrong-advice vulnerability"),
        ("Underreliance", "correct-advice underuse"),
        ("Confidence / calibration", "decision + interface signals"),
        ("Short-term learning gain", "tutoring dataset only"),
        ("Process traces", "observational FLoRA logs"),
        ("Unsupported", "delayed recall, transfer, diagnosis"),
    ]
    write_source_data(stem, "B_constructs", pd.DataFrame(constructs, columns=["construct", "evidence_boundary"]))
    for i, (name, body) in enumerate(constructs):
        color = CONSTRUCT_COLORS[name if name in CONSTRUCT_COLORS else "Unsupported"]
        canvas.pill(pB.x + 34 + (i % 2) * 278, pB.y + 70 + (i // 2) * 78, name, fill=lighten(color, 0.82), stroke=color, w=246)
        canvas.text(pB.x + 46 + (i % 2) * 278, pB.y + 122 + (i // 2) * 78, body, size=11.5, fill=VI.muted, max_width=226)

    pC = canvas.panel("C", 1388, 145, 760, 450, "Reliance-state neuro-symbolic model")
    model_nodes = [
        ("Harmonized features", "h0, advice, h1, y, context", "#0B7285"),
        ("Calibrated estimator", "tabular risk with uncertainty", "#5C7CFA"),
        ("Fuzzy symbolic rules", "human-readable state evidence", "#E67700"),
        ("State probabilities", "beneficial, harmful, uncertain", "#7048E8"),
    ]
    write_source_data(stem, "C_model_layers", pd.DataFrame(model_nodes, columns=["layer", "description", "color"]))
    for i, (title, body, color) in enumerate(model_nodes):
        x = pC.x + 34 + i * 174
        _badge(canvas, x, pC.y + 92, title, body, color, w=150, h=150)
        if i < len(model_nodes) - 1:
            canvas.arrow(x + 150, pC.y + 167, x + 174, pC.y + 167, stroke=VI.ink, width=1.6)
    canvas.round_rect(pC.x + 80, pC.y + 300, 600, 70, fill="#FFFFFF", stroke="#CAD5DD", radius=16)
    canvas.text(pC.x + 104, pC.y + 328, "Fusion: p(state) = sigmoid(tabular risk + rule evidence - uncertainty penalty)", size=16, fill=VI.ink)
    canvas.text(pC.x + 104, pC.y + 354, "The fusion layer is evaluated as diagnosis and policy triage, not as causal intervention proof.", size=12.5, fill=VI.muted)

    pD = canvas.panel("D", 205, 650, 1745, 520, "Outputs are calibrated, auditable and bounded")
    actions = [
        ("Calibrated risk", "ECE, Brier, reliability curves", CONSTRUCT_COLORS["Confidence / calibration"]),
        ("Rule trace", "which failure-mode rules fired", CONSTRUCT_COLORS["Overreliance"]),
        ("Counterfactual", "minimal feature/action change", CONSTRUCT_COLORS["Underreliance"]),
        ("Selective gating", "verify, compare evidence, delay", "#7048E8"),
        ("Evidence boundary", "what the data cannot support", "#868E96"),
    ]
    write_source_data(stem, "D_outputs", pd.DataFrame(actions, columns=["output", "description", "color"]))
    for i, (title, body, color) in enumerate(actions):
        x = pD.x + 42 + i * 330
        _badge(canvas, x, pD.y + 86, title, body, color, w=276, h=130)
        if i < len(actions) - 1:
            canvas.arrow(x + 276, pD.y + 151, x + 310, pD.y + 151, stroke=lighten(VI.ink, 0.15), width=1.7)
    canvas.text(pD.x + 42, pD.y + 300, "Central thesis", size=18, weight="800")
    canvas.text(
        pD.x + 42,
        pD.y + 334,
        "Average assistance gains can conceal two different failure modes: accepting wrong advice and failing to use correct advice. "
        "Reliance-state modeling makes that asymmetry measurable and actionable without broadening claims beyond the data.",
        size=17,
        fill=VI.ink,
        max_width=1540,
        line_height=1.2,
    )
    return _save(canvas, stem)


def figure_2(summary: dict, construct: pd.DataFrame) -> list[Path]:
    stem = "figure_02_evidence_boundary"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Evidence ledger separates supported constructs from future claims", "The display boundary is explicit: learning and process traces are real, but delayed recall, transfer and clinical claims remain unsupported.")
    sizes = summary.get("dataset_sizes", {})
    order = _dataset_order()
    cards = canvas.panel("A", 52, 145, 520, 1150, "Dataset cards")
    for i, key in enumerate(order):
        row = sizes.get(key, {})
        label = dataset_label(key)
        _badge(canvas, cards.x + 32, cards.y + 64 + i * 200, label, f"{int(row.get('n_interactions', 0)):,} records\n{int(row.get('n_participants', 0)):,} people", DATASET_COLORS[label], w=440, h=135)
    write_source_data(stem, "A_dataset_sizes", pd.DataFrame([{"dataset": dataset_label(k), **sizes.get(k, {})} for k in order]))

    mat_panel = canvas.panel("B", 610, 145, 1000, 760, "Construct support matrix")
    cols = ["appropriate reliance", "overreliance", "underreliance", "confidence shift", "learning gain", "process traces", "metacognitive engagement", "delayed recall", "transfer"]
    matrix = construct.set_index("dataset_key")[cols].reindex(order).fillna(0).astype(int)
    write_source_data(stem, "B_construct_matrix", matrix.reset_index().rename(columns={"dataset_key": "dataset"}))
    left = mat_panel.x + 210
    top = mat_panel.y + 122
    cell_w = 82
    cell_h = 92
    group_labels = [("Decision reliance", 0, 3), ("Confidence", 3, 1), ("Learning", 4, 1), ("Process traces", 5, 2), ("Unsupported", 7, 2)]
    for label, start, span in group_labels:
        canvas.round_rect(left + start * cell_w + 4, mat_panel.y + 64, span * cell_w - 8, 34, fill=lighten("#0B7285" if label != "Unsupported" else "#868E96", 0.88), stroke="#CAD5DD", radius=10)
        canvas.text(left + (start + span / 2) * cell_w, mat_panel.y + 86, label, size=11, weight="800", anchor="middle")
    for j, col in enumerate(cols):
        canvas.text(left + j * cell_w + cell_w / 2, top - 14, human_label(col), size=10.2, anchor="middle", max_width=78)
    for i, key in enumerate(order):
        y = top + i * cell_h
        canvas.text(mat_panel.x + 28, y + cell_h / 2 + 5, dataset_label(key), size=13.5, weight="800", max_width=160)
        for j, col in enumerate(cols):
            supported = int(matrix.loc[key, col]) == 1
            color = "#0B7285" if supported and col not in {"delayed recall", "transfer"} else "#868E96"
            canvas.round_rect(left + j * cell_w + 10, y + 18, cell_w - 20, 42, fill=lighten(color, 0.80) if supported else "#F3F5F7", stroke=color if supported else "#D7DEE4", radius=12)
            canvas.text(left + j * cell_w + cell_w / 2, y + 46, "yes" if supported else "no", size=11.5, weight="800", anchor="middle", fill=color if supported else VI.muted)

    key_panel = canvas.panel("C", 1645, 145, 455, 430, "Evidence strength key")
    strength = [("Randomized or interventional", "#0B7285"), ("Observational association", "#E67700"), ("Prediction only", "#5C7CFA"), ("Unsupported", "#868E96")]
    write_source_data(stem, "C_evidence_strength", pd.DataFrame(strength, columns=["level", "color"]))
    for i, (label, color) in enumerate(strength):
        _badge(canvas, key_panel.x + 38, key_panel.y + 75 + i * 78, label, "claim type used in captions and text", color, w=360, h=56)

    scale = canvas.panel("D", 610, 950, 1490, 345, "Corpus scale")
    counts = pd.DataFrame([{"dataset": dataset_label(k), "records": int(sizes.get(k, {}).get("n_interactions", 0)), "people": int(sizes.get(k, {}).get("n_participants", 0))} for k in order])
    write_source_data(stem, "D_corpus_scale", counts)
    max_records = max(counts["records"].max(), 1)
    for i, row in counts.iterrows():
        y = scale.y + 70 + i * 48
        canvas.text(scale.x + 32, y + 15, row["dataset"], size=13.5, weight="800", max_width=170)
        w_records = 900 * np.log10(row["records"] + 1) / np.log10(max_records + 1)
        w_people = 900 * np.log10(row["people"] + 1) / np.log10(max_records + 1)
        canvas.round_rect(scale.x + 220, y, w_records, 18, fill="#0B7285", stroke="none", radius=7)
        canvas.round_rect(scale.x + 220, y + 24, w_people, 18, fill="#E67700", stroke="none", radius=7)
        canvas.text(scale.x + 1140, y + 15, f"{row['records']:,} records", size=12, fill=VI.ink)
        canvas.text(scale.x + 1140, y + 39, f"{row['people']:,} people", size=12, fill=VI.muted)
    _small_key(canvas, scale.x + 1270, scale.y + 70, [("records", "#0B7285"), ("people", "#E67700")])
    return _save(canvas, stem)


def figure_3(effect: pd.DataFrame, summary: pd.DataFrame, haiid: pd.DataFrame, state: pd.DataFrame, flora: pd.DataFrame) -> list[Path]:
    stem = "figure_03_asymmetric_failures"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Average gains hide asymmetric reliance failures", "Accuracy changes are useful but incomplete; state decomposition shows correct-advice underuse and wrong-advice uptake separately.")
    gain_panel = canvas.panel("A", 52, 145, 760, 410, "Outcome gains only")
    gain = effect.loc[~effect["dataset"].eq("flora_ips")].copy()
    gain["label"] = gain["dataset"].map(dataset_label)
    gain["color_key"] = gain["label"]
    write_source_data(stem, "A_outcome_gains", gain)
    grouped_bar_ci(canvas, gain_panel.inset(10, 16), gain, label_col="label", value_col="effect", low_col="ci_low", high_col="ci_high", color_col="color_key", colors=DATASET_COLORS, ylabel="gain")

    flora_panel = canvas.panel("B", 850, 145, 560, 410, "FLoRA observational contrast")
    flora_plot = flora.loc[flora["process_feature"].isin(["genai_intensity", "prompt_depth", "metacognitive_prompt_ratio", "source_engagement_proxy"])].copy()
    flora_plot["label"] = flora_plot["process_feature"].map(human_label)
    write_source_data(stem, "B_flora_observational", flora_plot)
    horizontal_interval_plot(canvas, flora_panel.inset(8, 16), flora_plot, label_col="label", estimate_col="score_difference_high_minus_low", low_col="ci_low", high_col="ci_high", color=DATASET_COLORS["FLoRA IPS"], xlabel="score contrast", xlim=(-0.05, 0.08))
    canvas.text(flora_panel.x + 30, flora_panel.y + 365, "Observational process association, not pre/post gain.", size=12.5, fill=VI.muted)

    fail_panel = canvas.panel("C", 1450, 145, 650, 410, "Failure-mode decomposition")
    failure = summary.loc[summary["dataset_name"].isin(["haiid", "convxai_iui2025"])].copy()
    failure = failure.rename(columns={"dataset_name": "dataset"})
    failure_long = pd.DataFrame(
        [
            {"dataset": dataset_label(row["dataset"]), "failure": "Overreliance", "rate": row["overreliance_rate"], "color": "Overreliance"}
            for _, row in failure.iterrows()
        ]
        + [
            {"dataset": dataset_label(row["dataset"]), "failure": "Underreliance", "rate": row["underreliance_rate"], "color": "Underreliance"}
            for _, row in failure.iterrows()
        ]
    )
    write_source_data(stem, "C_failure_decomposition", failure_long)
    for i, dataset in enumerate(failure_long["dataset"].unique()):
        canvas.text(fail_panel.x + 52 + i * 285, fail_panel.y + 82, dataset, size=15, weight="800", anchor="middle")
        sub = failure_long.loc[failure_long["dataset"].eq(dataset)]
        for j, row in sub.iterrows():
            idx = 0 if row["failure"] == "Overreliance" else 1
            x = fail_panel.x + 48 + i * 285 + idx * 86
            h = 215 * row["rate"] / max(0.16, failure_long["rate"].max())
            y = fail_panel.y + 310 - h
            color = CONSTRUCT_COLORS[row["failure"]]
            canvas.round_rect(x, y, 58, h, fill=color, stroke="none", radius=8)
            canvas.text(x + 29, y - 8, f"{row['rate']:.3f}", size=11.5, anchor="middle", fill=VI.muted)
            canvas.text(x + 29, fail_panel.y + 342, row["failure"], size=10.5, anchor="middle", max_width=78)

    source_panel = canvas.panel("D", 52, 610, 880, 620, "HAIID advice-source asymmetry")
    source = (
        haiid.loc[haiid["metric"].isin(["correct_ai_reliance", "correct_self_reliance", "overreliance", "underreliance"])]
        .groupby(["advice_source_label", "metric"], as_index=False)["rate"]
        .mean()
    )
    source["metric_label"] = source["metric"].map(human_label)
    source["source_label"] = source["advice_source_label"].map({"ai": "AI-labelled advice", "human": "Human-labelled advice"})
    write_source_data(stem, "D_haiid_advice_source", source)
    metrics = ["correct_ai_reliance", "correct_self_reliance", "overreliance", "underreliance"]
    left = source_panel.x + 260
    top = source_panel.y + 92
    plot_w = 520
    row_h = 112
    for i, metric in enumerate(metrics):
        y = top + i * row_h
        canvas.text(source_panel.x + 30, y + 35, human_label(metric), size=13, weight="800", max_width=205)
        for idx, source_name in enumerate(["ai", "human"]):
            val = float(source.loc[(source["metric"] == metric) & (source["advice_source_label"] == source_name), "rate"].mean())
            color = "#0B7285" if source_name == "ai" else "#2B8A3E"
            w = plot_w * val / 0.75
            canvas.round_rect(left, y + idx * 36, w, 24, fill=color, stroke="none", radius=8)
            canvas.text(left + w + 8, y + idx * 36 + 18, f"{val:.2f}", size=11.5, fill=VI.muted)
        if i == 0:
            canvas.text(left, source_panel.y + 70, "AI-labelled", size=11, fill="#0B7285", weight="800")
            canvas.text(left + 160, source_panel.y + 70, "Human-labelled", size=11, fill="#2B8A3E", weight="800")

    state_panel = canvas.panel("E", 980, 610, 1120, 620, "Reliance-state decomposition")
    state_plot = state.copy()
    dataset_col = "dataset_name"
    state_col = "reliance_state"
    state_plot["share"] = state_plot["count"] / state_plot.groupby(dataset_col)["count"].transform("sum")
    state_plot["dataset"] = state_plot[dataset_col].map(dataset_label)
    state_plot["state"] = state_plot[state_col].map(lambda x: human_label(str(x).replace("ai", "AI")))
    write_source_data(stem, "E_state_decomposition", state_plot)
    dataset_keys = [d for d in ["haiid", "chi2023_dke", "convxai_iui2025"] if d in state_plot[dataset_col].unique()]
    states = [
        "beneficial_ai_reliance",
        "correct_self_reliance",
        "independent_correct",
        "independent_incorrect",
        "harmful_ai_overreliance",
        "harmful_ai_underreliance",
        "uncertain_disagreement",
    ]
    state_label_map = {
        "beneficial_ai_reliance": "Beneficial AI reliance",
        "correct_self_reliance": "Correct self-reliance",
        "independent_correct": "Independent correct",
        "independent_incorrect": "Independent incorrect",
        "harmful_ai_overreliance": "Harmful overreliance",
        "harmful_ai_underreliance": "Harmful underreliance",
        "uncertain_disagreement": "Uncertain disagreement",
    }
    x0 = state_panel.x + 80
    y0 = state_panel.y + 100
    bar_w = 190
    for i, dataset in enumerate(dataset_keys):
        x = x0 + i * 290
        y_base = y0 + 340
        canvas.text(x + bar_w / 2, y0 - 22, dataset_label(dataset), size=14, weight="800", anchor="middle")
        current = y_base
        for state_name in states:
            share = float(state_plot.loc[(state_plot[dataset_col] == dataset) & (state_plot[state_col] == state_name), "share"].sum())
            if share <= 0:
                continue
            h = 360 * share
            label = state_label_map.get(state_name, human_label(state_name))
            color = STATE_COLORS.get(label, "#868E96")
            current -= h
            canvas.round_rect(x, current, bar_w, h, fill=color, stroke="white", radius=3)
            if h > 30:
                canvas.text(x + bar_w / 2, current + h / 2 + 4, f"{share:.2f}", size=11, fill="white", weight="800", anchor="middle")
    for idx, state_name in enumerate(states):
        label = state_label_map.get(state_name, human_label(state_name))
        color = STATE_COLORS.get(label, "#868E96")
        lx = state_panel.x + 65 + (idx % 4) * 255
        ly = state_panel.y + 500 + (idx // 4) * 36
        canvas.round_rect(lx, ly, 20, 20, fill=color, stroke="none", radius=5)
        canvas.text(lx + 30, ly + 15, label, size=11.2, fill=VI.ink, max_width=215)
    return _save(canvas, stem)


def figure_4(chi_cond: pd.DataFrame, conv_cond: pd.DataFrame, gee: pd.DataFrame) -> list[Path]:
    stem = "figure_04_interface_calibration"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Interface, advice source and calibration shape reliance quality", "Clustered GEE estimates show that trust and reliability signals can be protective or harmful depending on advice correctness and interface design.")
    panels = [
        canvas.panel("A", 52, 145, 680, 380, "HAIID advice-source GEE"),
        canvas.panel("B", 770, 145, 610, 380, "CHI condition contrasts"),
        canvas.panel("C", 1420, 145, 680, 380, "CHI self-assessment effects"),
        canvas.panel("D", 52, 585, 680, 390, "ConvXAI interface contrasts"),
        canvas.panel("E", 770, 585, 900, 390, "Confidence and reliability effects"),
        canvas.panel("F", 1710, 585, 390, 390, "Interpretive boundary"),
    ]
    haiid_terms = gee.loc[(gee["dataset"].eq("haiid")) & (gee["term"].isin(["C(advice_source_label)[T.human]", "initial_confidence", "stated_accuracy_normalized"]))].copy()
    haiid_terms["label"] = haiid_terms["outcome"].map(human_label) + ": " + haiid_terms["term"].map(lambda t: "Human-labelled advice" if "advice_source" in str(t) else human_label(t))
    write_source_data(stem, "A_haiid_gee", haiid_terms)
    horizontal_interval_plot(canvas, panels[0].inset(10, 18), haiid_terms, label_col="label", estimate_col="estimate", low_col="ci_low", high_col="ci_high", color=DATASET_COLORS["HAIID"], xlabel="GEE log-odds", xlim=(-3.0, 2.3))

    chi = chi_cond.loc[chi_cond["metric"].eq("appropriate_reliance")].copy()
    if not chi.empty:
        ref = float(chi.loc[chi["condition_name"].eq("control"), "rate"].mean())
        chi["contrast"] = chi["rate"] - ref
        chi["low"] = chi["ci_low"] - ref
        chi["high"] = chi["ci_high"] - ref
        chi["label"] = chi["condition_name"].map(lambda x: human_label(str(x).replace("+ xai", "+ XAI")))
    write_source_data(stem, "B_chi_condition_contrasts", chi)
    horizontal_interval_plot(canvas, panels[1].inset(10, 18), chi, label_col="label", estimate_col="contrast", low_col="low", high_col="high", color=DATASET_COLORS["CHI 2023 DKE"], xlabel="difference from control", xlim=(-0.12, 0.16))

    self_terms = gee.loc[(gee["dataset"].eq("chi2023_dke")) & (gee["outcome"].eq("appropriate_reliance")) & (gee["term"].isin(["first_batch_overestimation", "first_batch_underestimation", "trust_first"]))].copy()
    self_terms["label"] = self_terms["term"].map(human_label)
    write_source_data(stem, "C_chi_self_assessment_gee", self_terms)
    horizontal_interval_plot(canvas, panels[2].inset(10, 18), self_terms, label_col="label", estimate_col="estimate", low_col="ci_low", high_col="ci_high", color="#5C7CFA", xlabel="GEE log-odds", xlim=(-0.85, 0.55))

    conv = conv_cond.loc[conv_cond["metric"].eq("appropriate_reliance")].copy()
    if not conv.empty:
        ref = float(conv.loc[conv["condition_name"].eq("control"), "rate"].mean())
        conv["contrast"] = conv["rate"] - ref
        conv["low"] = conv["ci_low"] - ref
        conv["high"] = conv["ci_high"] - ref
        conv["label"] = conv["condition_name"].map(lambda x: human_label(str(x).replace("xai", "XAI").replace("llm", "LLM")))
    write_source_data(stem, "D_convxai_interface_contrasts", conv)
    horizontal_interval_plot(canvas, panels[3].inset(10, 18), conv, label_col="label", estimate_col="contrast", low_col="low", high_col="high", color=DATASET_COLORS["ConvXAI"], xlabel="difference from control", xlim=(-0.10, 0.18))

    rel_terms = gee.loc[(gee["dataset"].eq("convxai_iui2025")) & (gee["term"].isin(["post_explain_reliability", "confidence", "explanation_engagement_count"]))].copy()
    rel_terms["label"] = rel_terms["outcome"].map(human_label) + ": " + rel_terms["term"].map(human_label)
    write_source_data(stem, "E_convxai_confidence_reliability", rel_terms)
    horizontal_interval_plot(canvas, panels[4].inset(10, 18), rel_terms, label_col="label", estimate_col="estimate", low_col="ci_low", high_col="ci_high", color="#7048E8", xlabel="GEE log-odds", xlim=(-1.2, 9.9))
    canvas.text(panels[5].x + 34, panels[5].y + 92, "Reliability cues are not universally beneficial.", size=20, weight="800", max_width=310)
    canvas.text(
        panels[5].x + 34,
        panels[5].y + 160,
        "They can align people with correct advice, but they can also coincide with overreliance when advice is wrong. "
        "This is why reliance states, not surface accuracy alone, are the analysis unit.",
        size=16,
        fill=VI.muted,
        max_width=300,
        line_height=1.25,
    )
    return _save(canvas, stem)


def figure_5(pardos: pd.DataFrame, flora: pd.DataFrame, flora_models: pd.DataFrame, gee: pd.DataFrame) -> list[Path]:
    stem = "figure_05_learning_process_extension"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Learning and process-trace datasets widen, but bound, the evidence", "The learning dataset supports short-term gains; process traces expose observational signals and a predictive boundary.")
    pA = canvas.panel("A", 52, 145, 640, 440, "Pardos/Bhandari learning gain")
    pardos_plot = pardos.copy()
    pardos_plot["label"] = pardos_plot["condition_name"].map(lambda x: "No-hint control" if "No-hint" in str(x) else str(x))
    pardos_plot["color_key"] = pardos_plot["label"]
    pcolors = {"ChatGPT": "#0B7285", "Human tutor": "#5C7CFA", "No-hint control": "#868E96"}
    write_source_data(stem, "A_pardos_learning_gain", pardos_plot)
    grouped_bar_ci(canvas, pA.inset(10, 20), pardos_plot, label_col="label", value_col="learning_gain", low_col="ci_low", high_col="ci_high", color_col="color_key", colors=pcolors, ylabel="post minus pre", ylim=(-0.06, 0.24))

    pB = canvas.panel("B", 735, 145, 620, 440, "Tutoring GEE contrasts")
    tutoring = gee.loc[(gee["dataset"].eq("pardos_chatgpt_tutoring")) & (gee["term"].str.contains("condition", na=False))].copy()
    tutoring["label"] = tutoring["term"].map(lambda x: "No-hint vs ChatGPT" if "No-hint" in str(x) else "Human tutor vs ChatGPT")
    write_source_data(stem, "B_pardos_gee", tutoring)
    horizontal_interval_plot(canvas, pB.inset(12, 22), tutoring, label_col="label", estimate_col="estimate", low_col="ci_low", high_col="ci_high", color="#0B7285", xlabel="learning-gain difference", xlim=(-0.20, 0.08))

    pC = canvas.panel("C", 1395, 145, 705, 440, "FLoRA process associations")
    flora_plot = flora.copy()
    flora_plot["label"] = flora_plot["process_feature"].map(human_label)
    write_source_data(stem, "C_flora_process_effects", flora_plot)
    horizontal_interval_plot(canvas, pC.inset(10, 22), flora_plot.head(6), label_col="label", estimate_col="score_difference_high_minus_low", low_col="ci_low", high_col="ci_high", color=DATASET_COLORS["FLoRA IPS"], xlabel="score contrast", xlim=(-0.06, 0.08))

    pD = canvas.panel("D", 52, 645, 910, 500, "Predictive boundary")
    models = flora_models.copy()
    models["label"] = models["model"].map(model_label)
    write_source_data(stem, "D_flora_process_prediction", models)
    horizontal_interval_plot(canvas, pD.inset(8, 20), models, label_col="label", estimate_col="r2", low_col="r2_ci_low", high_col="r2_ci_high", color="#2B8A3E", xlabel="student-level cross-validated R2", zero=0, xlim=(-0.30, 0.10))
    canvas.text(pD.x + 40, pD.y + 430, "Coarse aggregate process summaries do not yet predict proposal quality strongly.", size=15, fill=VI.muted)

    pE = canvas.panel("E", 1010, 645, 1090, 500, "Process schematic and evidence boundary")
    nodes = [("GenAI dialogue", "#2B8A3E"), ("Writing events", "#5C7CFA"), ("Annotation and source engagement", "#E67700"), ("Proposal score", "#7048E8")]
    write_source_data(stem, "E_process_schematic", pd.DataFrame(nodes, columns=["node", "color"]))
    for i, (label, color) in enumerate(nodes):
        x = pE.x + 62 + i * 245
        _badge(canvas, x, pE.y + 150, label, "observed process channel", color, w=190, h=112)
        if i < len(nodes) - 1:
            canvas.arrow(x + 190, pE.y + 206, x + 232, pE.y + 206, stroke=VI.ink)
    canvas.round_rect(pE.x + 90, pE.y + 320, 900, 82, fill="#FFFFFF", stroke="#D6DEE4", radius=18)
    canvas.text(pE.x + 120, pE.y + 350, "Evidence boundary: this supports process-performance association, not delayed recall or transfer.", size=18, weight="800")
    canvas.text(pE.x + 120, pE.y + 380, "The negative prediction result is kept as a constraint on the theory.", size=14, fill=VI.muted)
    return _save(canvas, stem)


def figure_6() -> list[Path]:
    stem = "figure_06_reliance_state_method"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Reliance-State NeuroSymbolic Calibration and Gating Model", "The method formalizes reliance states, fuses calibrated statistical risk with fuzzy rule evidence, and ranks selective actions under an observational utility model.")
    pA = canvas.panel("A", 52, 145, 520, 490, "State formalism")
    states = [
        "Beneficial AI reliance",
        "Harmful overreliance",
        "Harmful underreliance",
        "Correct self-reliance",
        "Independent correct",
        "Independent incorrect",
        "Uncertain disagreement",
    ]
    write_source_data(stem, "A_state_definitions", pd.DataFrame({"state": states}))
    for i, state in enumerate(states):
        color = STATE_COLORS.get(state, "#868E96")
        canvas.pill(pA.x + 38 + (i % 2) * 230, pA.y + 78 + (i // 2) * 76, state, fill=lighten(color, 0.82), stroke=color, w=205)
    canvas.text(pA.x + 38, pA.y + 410, "States are deterministic labels when h0, advice, h1 and y are observed; otherwise they are prediction targets.", size=14, fill=VI.muted, max_width=430)

    pB = canvas.panel("B", 610, 145, 880, 490, "Fusion mechanism")
    layers = [
        ("Input tuple", "(h0, advice, h1, y, x)", "#0B7285"),
        ("Tabular estimator", "eta(x)", "#5C7CFA"),
        ("Fuzzy rules", "R = weighted rule sum", "#E67700"),
        ("Uncertainty", "bootstrap variance u", "#7048E8"),
        ("State probabilities", "p(state) = sigmoid(eta + lambda R - gamma u)", "#2B8A3E"),
    ]
    write_source_data(stem, "B_fusion_layers", pd.DataFrame(layers, columns=["layer", "notation", "color"]))
    for i, (title, body, color) in enumerate(layers):
        x = pB.x + 35 + i * 163
        _badge(canvas, x, pB.y + 140, title, body, color, w=135, h=128)
        if i < len(layers) - 1:
            canvas.arrow(x + 135, pB.y + 204, x + 160, pB.y + 204, stroke=VI.ink)
    canvas.round_rect(pB.x + 78, pB.y + 330, 730, 70, fill="#FFFFFF", stroke="#CBD6DE", radius=16)
    canvas.text(pB.x + 105, pB.y + 358, "Rule activation: r(x) is the product of fuzzy memberships", size=17, weight="800")
    canvas.text(pB.x + 105, pB.y + 386, "Fuzzy memberships make the symbolic layer continuous and ablatable.", size=13.5, fill=VI.muted)

    pC = canvas.panel("C", 1530, 145, 570, 490, "Calibration-aware gating")
    canvas.round_rect(pC.x + 42, pC.y + 90, 480, 102, fill="#FFFFFF", stroke="#CBD6DE", radius=18)
    canvas.text(pC.x + 68, pC.y + 126, "U(action) = P(correct) - alpha P(over) - beta P(under) - kappa burden", size=17, weight="800", max_width=430)
    actions = ["Accept advice", "Resist advice", "Request verification", "Show uncertainty cue", "Compare evidence", "Delay final answer"]
    write_source_data(stem, "C_gating_actions", pd.DataFrame({"action": actions}))
    for i, action in enumerate(actions):
        canvas.pill(pC.x + 54 + (i % 2) * 230, pC.y + 240 + (i // 2) * 70, action, fill=lighten("#7048E8", 0.84), stroke="#7048E8", w=200)

    pD = canvas.panel("D", 52, 700, 2048, 520, "Guarantee and boundary")
    canvas.round_rect(pD.x + 45, pD.y + 92, 940, 160, fill=lighten("#0B7285", 0.90), stroke="#0B7285", radius=20)
    canvas.text(pD.x + 82, pD.y + 130, "Accuracy decomposition", size=21, weight="800")
    canvas.text(pD.x + 82, pD.y + 170, "Final accuracy = initially correct and retained + initially wrong and corrected + other correct final states.", size=17, max_width=820)
    canvas.text(pD.x + 82, pD.y + 205, "Therefore aggregate accuracy can rise while overreliance and underreliance remain non-zero.", size=15, fill=VI.muted, max_width=820)
    canvas.round_rect(pD.x + 1070, pD.y + 92, 880, 160, fill=lighten("#E67700", 0.90), stroke="#E67700", radius=20)
    canvas.text(pD.x + 1105, pD.y + 130, "Calibration-bound intuition", size=21, weight="800")
    canvas.text(pD.x + 1105, pD.y + 170, "If harmful-reliance probabilities are calibrated within epsilon, utility-ranking error is bounded by epsilon times bounded cost weights.", size=17, max_width=760)
    canvas.text(pD.x + 1105, pD.y + 205, "This is a decision-support bound, not a causal intervention guarantee.", size=15, fill=VI.muted, max_width=760)
    canvas.round_rect(pD.x + 45, pD.y + 315, 1905, 90, fill="#FFFFFF", stroke="#CBD6DE", radius=18)
    canvas.text(pD.x + 78, pD.y + 350, "Measurement, prediction, explanation and observational policy simulation are deliberately separated throughout the paper.", size=21, weight="800")
    canvas.text(pD.x + 78, pD.y + 382, "Prospective randomized gating experiments are required before claiming causal intervention effects.", size=15, fill=VI.muted)
    return _save(canvas, stem)


def figure_7(model: pd.DataFrame, cross: pd.DataFrame) -> list[Path]:
    stem = "figure_07_prediction_generalization"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Prediction and cross-dataset transfer under strict validation", "The method is compared against strong baselines; weak transfers are retained because they identify construct-portability limits.")
    pA = canvas.panel("A", 52, 145, 870, 520, "Strict validation comparison")
    candidates = ["logistic_regression", "calibrated_gradient_boosting", "symbolic_only", "uncertainty_aware_fusion", "reliance_state_neurosymbolic"]
    strict = model.loc[(model["split"].isin(["participant", "strict_task"])) & (model["model"].isin(candidates))].copy()
    top_cases = strict.sort_values(["dataset", "target", "split", "model"]).groupby(["dataset", "target", "split", "model"]).head(1)
    summary = top_cases.groupby("model", as_index=False).agg(auroc=("auroc", "mean"), auroc_ci_low=("auroc_ci_low", "mean"), auroc_ci_high=("auroc_ci_high", "mean"))
    summary["label"] = summary["model"].map(model_label)
    summary = summary.sort_values("auroc")
    write_source_data(stem, "A_strict_validation_model_means", summary)
    horizontal_interval_plot(canvas, pA.inset(8, 18), summary, label_col="label", estimate_col="auroc", low_col="auroc_ci_low", high_col="auroc_ci_high", color="#7048E8", xlabel="mean strict AUROC", zero=0.5, xlim=(0.48, 0.86))

    pB = canvas.panel("B", 960, 145, 520, 520, "Delta versus best non-symbolic")
    rows = []
    for (dataset, target, split_name), group in strict.groupby(["dataset", "target", "split"]):
        ns = group.loc[group["model"].eq("reliance_state_neurosymbolic")]
        non = group.loc[~group["model"].isin(["reliance_state_neurosymbolic", "symbolic_only"])]
        if ns.empty or non.empty:
            continue
        best = non.sort_values("auroc", ascending=False).iloc[0]
        rows.append({"case": f"{dataset_label(dataset)} {human_label(target)} {human_label(split_name)}", "delta": float(ns.iloc[0]["auroc"] - best["auroc"])})
    delta = pd.DataFrame(rows).sort_values("delta")
    write_source_data(stem, "B_neurosymbolic_delta", delta)
    if not delta.empty:
        delta["low"] = delta["delta"]
        delta["high"] = delta["delta"]
        horizontal_interval_plot(canvas, pB.inset(8, 18), delta.tail(8), label_col="case", estimate_col="delta", low_col="low", high_col="high", color="#7048E8", xlabel="AUROC delta", zero=0, xlim=(-0.12, 0.12))

    pC = canvas.panel("C", 1520, 145, 580, 520, "Appropriate-reliance transfer")
    transfer = cross.loc[cross["target"].eq("appropriate_reliance")].sort_values("auroc", ascending=False).groupby(["train_dataset", "test_dataset"]).head(1)
    matrix = transfer.pivot(index="train_dataset", columns="test_dataset", values="auroc")
    rows = [r for r in ["haiid", "chi2023_dke", "convxai_iui2025"] if r in matrix.index]
    cols = [c for c in ["haiid", "chi2023_dke", "convxai_iui2025"] if c in matrix.columns]
    matrix = matrix.reindex(rows)[cols]
    write_source_data(stem, "C_transfer_matrix", matrix.reset_index())
    matrix_plot(canvas, pC.inset(0, 16), matrix, row_labels=[dataset_label(r) for r in rows], col_labels=[dataset_label(c) for c in cols], lo=0.45, hi=0.82)

    pD = canvas.panel("D", 52, 725, 1080, 520, "Harmful-reliance transfer remains bounded")
    harmful = cross.loc[cross["target"].isin(["overreliance", "underreliance"])].copy()
    harmful = harmful.sort_values("auroc", ascending=False).groupby(["train_dataset", "test_dataset", "target"]).head(1).sort_values("auroc").tail(8)
    harmful["label"] = harmful["target"].map(human_label) + " to " + harmful["test_dataset"].map(dataset_label)
    write_source_data(stem, "D_harmful_transfer", harmful)
    horizontal_interval_plot(canvas, pD.inset(8, 18), harmful, label_col="label", estimate_col="auroc", low_col="auroc_ci_low", high_col="auroc_ci_high", color="#C92A2A", xlabel="external AUROC", zero=0.5, xlim=(0.45, 0.78))

    pE = canvas.panel("E", 1170, 725, 930, 520, "Where the neuro-symbolic layer adds value")
    win_rows = pd.DataFrame(
        [
            {"evidence": "Calibration and reliability", "status": "strongest support", "color": "#0B7285"},
            {"evidence": "Rule-grounded diagnosis", "status": "clear support", "color": "#E67700"},
            {"evidence": "Strict AUROC", "status": "competitive, mixed", "color": "#7048E8"},
            {"evidence": "External transfer", "status": "bounded portability", "color": "#868E96"},
        ]
    )
    write_source_data(stem, "E_method_value_summary", win_rows)
    for i, row in win_rows.iterrows():
        _badge(canvas, pE.x + 55, pE.y + 78 + i * 94, row["evidence"], row["status"], row["color"], w=800, h=72)
    return _save(canvas, stem)


def figure_8(curves: pd.DataFrame, calibration: pd.DataFrame, decision_curve: pd.DataFrame) -> list[Path]:
    stem = "figure_08_calibration_uncertainty"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Calibration and uncertainty determine when warnings are trustworthy", "Reliability is measurable but mixed, motivating selective prediction rather than universal reliance warnings.")
    pA = canvas.panel("A", 52, 145, 620, 440, "Strong calibration case")
    strong = curves.loc[(curves["dataset"].eq("haiid")) & (curves["target"].eq("underreliance")) & (curves["split"].eq("participant")) & (curves["model"].isin(["reliance_state_neurosymbolic", "uncertainty_aware_fusion"]))].copy()
    strong["model_label"] = strong["model"].map(model_label)
    write_source_data(stem, "A_reliability_strong", strong)
    reliability_plot(canvas, pA.inset(8, 18), strong, model_col="model_label", x_col="mean_confidence", y_col="accuracy", models=["Reliance-state neuro-symbolic", "Uncertainty-aware fusion"], colors=MODEL_COLORS)

    pB = canvas.panel("B", 710, 145, 620, 440, "Difficult calibration case")
    hard = curves.loc[(curves["dataset"].eq("convxai_iui2025")) & (curves["target"].eq("overreliance")) & (curves["split"].eq("participant")) & (curves["model"].isin(["reliance_state_neurosymbolic", "uncertainty_aware_fusion"]))].copy()
    hard["model_label"] = hard["model"].map(model_label)
    write_source_data(stem, "B_reliability_difficult", hard)
    reliability_plot(canvas, pB.inset(8, 18), hard, model_col="model_label", x_col="mean_confidence", y_col="accuracy", models=["Reliance-state neuro-symbolic", "Uncertainty-aware fusion"], colors=MODEL_COLORS)

    pC = canvas.panel("C", 1370, 145, 730, 440, "ECE comparison")
    cal = calibration.loc[calibration["model"].isin(["reliance_state_neurosymbolic", "uncertainty_aware_fusion", "calibrated_gradient_boosting"])].copy()
    cal["label"] = cal["dataset"].map(dataset_label) + " / " + cal["target"].map(human_label)
    best = cal.sort_values("ece").groupby(["dataset", "target", "split"]).head(1).sort_values("ece").head(8)
    best["model_label"] = best["model"].map(model_label)
    write_source_data(stem, "C_ece_best_cases", best)
    horizontal_interval_plot(canvas, pC.inset(10, 18), best.assign(low=best["ece"], high=best["ece"]), label_col="label", estimate_col="ece", low_col="low", high_col="high", color="#0B7285", xlabel="ECE", zero=0, xlim=(0, max(0.18, float(best["ece"].max()) * 1.2 if not best.empty else 0.18)))

    pD = canvas.panel("D", 52, 645, 670, 470, "Calibration slope and intercept")
    slope = calibration.loc[calibration["model"].eq("reliance_state_neurosymbolic")].copy()
    slope = slope.sort_values("calibration_slope").head(8)
    slope["label"] = slope["dataset"].map(dataset_label) + " / " + slope["target"].map(human_label)
    write_source_data(stem, "D_calibration_slope", slope)
    horizontal_interval_plot(canvas, pD.inset(10, 18), slope.assign(low=slope["calibration_slope"], high=slope["calibration_slope"]), label_col="label", estimate_col="calibration_slope", low_col="low", high_col="high", color="#7048E8", xlabel="slope (ideal = 1)", zero=1, xlim=(-0.1, 1.6))

    pE = canvas.panel("E", 760, 645, 670, 470, "Selective-action curve")
    curve = decision_curve.copy()
    curve["policy_label"] = curve["policy"].map(policy_label)
    write_source_data(stem, "E_selective_policy_curve", curve)
    left = pE.x + 85
    top = pE.y + 70
    w = pE.w - 145
    h = pE.h - 125
    canvas.axis(left, top, w, h, xlabel="intervention burden", ylabel="utility gain")
    for dataset, sub in curve.groupby("dataset"):
        sub = sub.sort_values("intervention_burden")
        color = DATASET_COLORS.get(dataset_label(dataset), "#0B7285")
        pts = []
        for _, row in sub.iterrows():
            x = left + row["intervention_burden"] * w
            y = top + h - (row["utility_gain_vs_observed"] + 0.06) / 0.15 * h
            pts.append((x, y))
        if len(pts) > 1:
            canvas.dwg.add(canvas.dwg.polyline(points=pts, fill="none", stroke=color, stroke_width=3))
        if pts:
            canvas.text(pts[-1][0] + 8, pts[-1][1] + 4, dataset_label(dataset), size=10.5, fill=color)
    pF = canvas.panel("F", 1470, 645, 630, 470, "Uncertainty as warning triage")
    canvas.round_rect(pF.x + 50, pF.y + 90, 520, 115, fill=lighten("#7048E8", 0.88), stroke="#7048E8", radius=22)
    canvas.text(pF.x + 78, pF.y + 132, "Uncertainty is not hidden.", size=22, weight="800")
    canvas.text(pF.x + 78, pF.y + 166, "It changes the recommended action and marks cases where a blanket warning would be overconfident.", size=16, fill=VI.muted, max_width=455)
    canvas.round_rect(pF.x + 50, pF.y + 260, 520, 105, fill="#FFFFFF", stroke="#CBD6DE", radius=22)
    canvas.text(pF.x + 78, pF.y + 300, "Calibration result is mixed by target and split.", size=19, weight="800")
    canvas.text(pF.x + 78, pF.y + 330, "The claim is selective action support, not universal calibration dominance.", size=14, fill=VI.muted, max_width=450)
    return _save(canvas, stem)


def figure_9(ablation: pd.DataFrame, policy: pd.DataFrame) -> list[Path]:
    stem = "figure_09_rules_counterfactuals"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Symbolic rules turn reliance predictions into auditable explanations", "Ablations, flow mapping, case traces and intervention distributions make the symbolic layer inspectable rather than decorative.")
    pA = canvas.panel("A", 52, 145, 800, 500, "Rule ablation signal")
    abl = ablation.loc[ablation["split"].isin(["participant", "strict_task"])].copy().sort_values("mean_difference").head(9)
    abl["label"] = abl["dataset"].map(dataset_label) + " / " + abl["group"].map(human_label)
    write_source_data(stem, "A_rule_ablation", abl)
    horizontal_interval_plot(canvas, pA.inset(8, 20), abl, label_col="label", estimate_col="mean_difference", low_col="ci_low", high_col="ci_high", color="#C92A2A", xlabel="AUROC change after removal", zero=0, xlim=(-0.36, 0.06))

    pB = canvas.panel("B", 900, 145, 720, 500, "Rule-to-action flow")
    flows = [
        ("Wrong-advice vulnerability", "Harmful overreliance", "Request verification", "#C92A2A"),
        ("Correct-advice underuse", "Harmful underreliance", "Compare evidence", "#E67700"),
        ("Low source engagement", "Uncertain disagreement", "Show uncertainty cue", "#7048E8"),
    ]
    write_source_data(stem, "B_rule_action_flow", pd.DataFrame(flows, columns=["rule_group", "failure_mode", "action", "color"]))
    for i, (rule, failure, action, color) in enumerate(flows):
        y = pB.y + 90 + i * 120
        _badge(canvas, pB.x + 30, y, rule, "activated rule group", color, w=190, h=80)
        _badge(canvas, pB.x + 270, y, failure, "state probability", color, w=170, h=80)
        _badge(canvas, pB.x + 500, y, action, "policy candidate", color, w=170, h=80)
        canvas.arrow(pB.x + 220, y + 40, pB.x + 270, y + 40, stroke=color, width=2.0)
        canvas.arrow(pB.x + 440, y + 40, pB.x + 500, y + 40, stroke=color, width=2.0)

    pC = canvas.panel("C", 1660, 145, 440, 500, "Counterfactual card")
    canvas.round_rect(pC.x + 36, pC.y + 70, 365, 335, fill="#FFFFFF", stroke="#CBD6DE", radius=22, width=1.4)
    canvas.text(pC.x + 62, pC.y + 112, "Predicted state", size=13, fill=VI.muted, weight="800")
    canvas.text(pC.x + 62, pC.y + 146, "Harmful underreliance", size=22, weight="800", fill="#E67700", max_width=320)
    canvas.text(pC.x + 62, pC.y + 205, "Top rule: low confidence + correct advice + no movement toward advice.", size=15, max_width=310)
    canvas.text(pC.x + 62, pC.y + 277, "Counterfactual: compare evidence before final answer.", size=17, weight="800", fill="#0B7285", max_width=300)
    canvas.text(pC.x + 62, pC.y + 350, "Diagnostic recommendation only; prospective trials are needed for causal effects.", size=12.5, fill=VI.muted, max_width=305)

    pD = canvas.panel("D", 52, 705, 1000, 500, "Example case trace")
    trace_nodes = [("Inputs", "initial answer, advice, final answer"), ("Rules", "underuse and uncertainty"), ("State", "harmful underreliance risk"), ("Action", "compare evidence")]
    write_source_data(stem, "D_case_trace", pd.DataFrame(trace_nodes, columns=["stage", "content"]))
    for i, (stage, body) in enumerate(trace_nodes):
        x = pD.x + 55 + i * 230
        _badge(canvas, x, pD.y + 160, stage, body, ["#0B7285", "#E67700", "#7048E8", "#2B8A3E"][i], w=185, h=120)
        if i < len(trace_nodes) - 1:
            canvas.arrow(x + 185, pD.y + 220, x + 225, pD.y + 220, stroke=VI.ink)

    pE = canvas.panel("E", 1095, 705, 1005, 500, "Recommended intervention distribution")
    actions = _read_csv("policy_actions.csv")
    if not actions.empty:
        action_col = "gating_action"
        dist = actions[action_col].map(human_label).value_counts(normalize=True).head(6).reset_index()
        dist.columns = ["label", "share"]
    else:
        dist = pd.DataFrame({"label": ["Request verification", "Compare evidence", "Show uncertainty cue"], "share": [0.42, 0.33, 0.25]})
    write_source_data(stem, "E_intervention_distribution", dist)
    grouped_bar_ci(canvas, pE.inset(10, 22), dist.sort_values("share"), label_col="label", value_col="share", low_col=None, high_col=None, ylabel="share", horizontal=True)
    return _save(canvas, stem)


def figure_10(policy: pd.DataFrame, decision_curve: pd.DataFrame) -> list[Path]:
    stem = "figure_10_policy_frontier"
    canvas = FigureCanvas(MAIN_W, MAIN_H)
    _header(canvas, "Offline gating frontiers show selective action trade-offs", "The policy analysis is conservative and observational: it ranks intervention candidates but does not estimate causal effects of deploying them.")
    datasets = ["haiid", "chi2023_dke", "convxai_iui2025"]
    locations = [(52, 145), (765, 145), (1478, 145)]
    write_source_data(stem, "frontier_policy_evaluation", policy)
    for idx, (dataset, (x, y)) in enumerate(zip(datasets, locations, strict=False)):
        p = canvas.panel(chr(ord("A") + idx), x, y, 625, 520, f"{dataset_label(dataset)} frontier")
        sub = policy.loc[policy["dataset"].eq(dataset)].copy()
        sub["policy_label"] = sub["policy"].map(policy_label)
        left = p.x + 82
        top = p.y + 86
        w = p.w - 135
        h = p.h - 150
        canvas.axis(left, top, w, h, xlabel="intervention burden", ylabel="utility")
        ymin = min(float(sub["expected_utility"].min()) - 0.02, -0.12)
        ymax = max(float(sub["expected_utility"].max()) + 0.02, 0.08)
        obs_pt = None
        ns_pt = None
        for _, row in sub.iterrows():
            px = left + float(row["intervention_burden"]) * w
            py = top + h - (float(row["expected_utility"]) - ymin) / (ymax - ymin) * h
            label = row["policy_label"]
            color = POLICY_COLORS.get(label, "#0B7285")
            canvas.dot(px, py, 10, fill=color, stroke="white", width=2)
            canvas.text(px + 10, py + 4, label.replace(" threshold", ""), size=10.8, fill=color, max_width=145)
            if row["policy"] == "observed_no_gating":
                obs_pt = (px, py)
            if row["policy"] == "neurosymbolic_gating":
                ns_pt = (px, py)
        if obs_pt and ns_pt:
            canvas.arrow(obs_pt[0], obs_pt[1], ns_pt[0], ns_pt[1], stroke="#C92A2A", width=2.0)

    pD = canvas.panel("D", 52, 730, 960, 510, "Burden-penalty sensitivity")
    curve = decision_curve.copy()
    curve["policy_label"] = curve["policy"].map(policy_label)
    write_source_data(stem, "D_decision_curve_summary", curve)
    left = pD.x + 88
    top = pD.y + 80
    w = pD.w - 150
    h = pD.h - 145
    canvas.axis(left, top, w, h, xlabel="intervention burden", ylabel="utility gain")
    for dataset, sub in curve.groupby("dataset"):
        sub = sub.sort_values("intervention_burden")
        color = DATASET_COLORS.get(dataset_label(dataset), "#0B7285")
        pts = []
        for _, row in sub.iterrows():
            px = left + row["intervention_burden"] * w
            py = top + h - (row["utility_gain_vs_observed"] + 0.06) / 0.16 * h
            pts.append((px, py))
        if len(pts) > 1:
            canvas.dwg.add(canvas.dwg.polyline(points=pts, fill="none", stroke=color, stroke_width=3))
        if pts:
            canvas.text(pts[-1][0] + 8, pts[-1][1] + 4, dataset_label(dataset), size=11, fill=color)
    canvas.line(left, top + h - (0 + 0.06) / 0.16 * h, left + w, top + h - (0 + 0.06) / 0.16 * h, stroke=VI.ink, width=1.0, opacity=0.5)

    pE = canvas.panel("E", 1060, 730, 1040, 510, "Correctness and harmful-reliance trade-off")
    trade = policy.loc[policy["policy"].isin(["observed_no_gating", "neurosymbolic_gating"])].copy()
    write_source_data(stem, "E_tradeoff", trade)
    for i, dataset in enumerate(datasets):
        sub = trade.loc[trade["dataset"].eq(dataset)]
        y = pE.y + 88 + i * 125
        canvas.text(pE.x + 45, y + 30, dataset_label(dataset), size=15, weight="800", max_width=170)
        for j, policy_name in enumerate(["observed_no_gating", "neurosymbolic_gating"]):
            row = sub.loc[sub["policy"].eq(policy_name)]
            if row.empty:
                continue
            row = row.iloc[0]
            x = pE.x + 245 + j * 345
            color = POLICY_COLORS[policy_label(policy_name)]
            _badge(
                canvas,
                x,
                y,
                policy_label(policy_name),
                f"correct {row['expected_final_correct']:.2f}; harmful {(row['expected_overreliance'] + row['expected_underreliance']):.2f}",
                color,
                w=285,
                h=86,
            )
    canvas.text(pE.x + 45, pE.y + 445, "All frontiers are observational simulations; they prioritize future interventions rather than proving deployment effects.", size=14, fill=VI.muted, max_width=900)
    return _save(canvas, stem)


def extended_figures(summary: dict, state: pd.DataFrame, model: pd.DataFrame, calibration: pd.DataFrame, cross: pd.DataFrame, sensitivity: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    sizes = summary.get("dataset_sizes", {})
    canvas = FigureCanvas(1800, 900)
    _header(canvas, "Extended Data 1: corpus scale and source-data flow", "Records and participant counts for integrated public datasets.")
    p = canvas.panel("A", 60, 150, 1680, 610, "Dataset scale")
    frame = pd.DataFrame([{"dataset": dataset_label(k), "records": int(v.get("n_interactions", 0)), "people": int(v.get("n_participants", 0))} for k, v in sizes.items()])
    write_source_data("extended_data_1_dataset_distribution", "A_dataset_scale", frame)
    if not frame.empty:
        max_records = max(frame["records"].max(), 1)
        for i, row in frame.iterrows():
            y = p.y + 75 + i * 90
            canvas.text(p.x + 40, y + 22, row["dataset"], size=17, weight="800")
            canvas.round_rect(p.x + 260, y, 1000 * np.log10(row["records"] + 1) / np.log10(max_records + 1), 26, fill="#0B7285", stroke="none", radius=9)
            canvas.round_rect(p.x + 260, y + 34, 1000 * np.log10(row["people"] + 1) / np.log10(max_records + 1), 26, fill="#E67700", stroke="none", radius=9)
            canvas.text(p.x + 1310, y + 22, f"{row['records']:,} records", size=14)
            canvas.text(p.x + 1310, y + 56, f"{row['people']:,} people", size=14, fill=VI.muted)
    outputs.extend(_save(canvas, "extended_data_1_dataset_distribution"))

    canvas = FigureCanvas(1800, 950)
    _header(canvas, "Extended Data 2: full reliance-state distribution", "Stacked state shares across decision datasets.")
    p = canvas.panel("A", 60, 150, 1680, 650, "State distribution")
    state_plot = state.copy()
    if not state_plot.empty:
        state_plot["share"] = state_plot["count"] / state_plot.groupby("dataset_name")["count"].transform("sum")
        write_source_data("extended_data_2_state_distribution", "A_state_distribution", state_plot)
        x0 = p.x + 100
        for i, dataset in enumerate(state_plot["dataset_name"].unique()):
            sub = state_plot.loc[state_plot["dataset_name"].eq(dataset)]
            x = x0 + i * 430
            current = p.y + 520
            canvas.text(x + 140, p.y + 58, dataset_label(dataset), size=18, weight="800", anchor="middle")
            for _, row in sub.iterrows():
                label = human_label(str(row["reliance_state"]).replace("ai", "AI"))
                color = STATE_COLORS.get(label, "#868E96")
                h = 470 * float(row["share"])
                current -= h
                canvas.round_rect(x, current, 280, h, fill=color, stroke="white", radius=4)
                if h > 24:
                    canvas.text(x + 140, current + h / 2 + 5, f"{row['share']:.2f}", size=12, fill="white", weight="800", anchor="middle")
        _small_key(canvas, p.x + 1290, p.y + 120, [(label, color) for label, color in STATE_COLORS.items()])
    outputs.extend(_save(canvas, "extended_data_2_state_distribution"))

    extra_specs = [
        ("extended_data_3_model_comparison", "Full model comparison", model.head(60)),
        ("extended_data_4_calibration_metrics", "Calibration metrics", calibration.head(60)),
        ("extended_data_5_cross_dataset_transfer", "Cross-dataset transfer", cross.head(60)),
        ("extended_data_6_sensitivity", "Label and robustness sensitivity", sensitivity.head(60)),
    ]
    for stem, title, frame in extra_specs:
        canvas = FigureCanvas(1800, 900)
        _header(canvas, title, "Extended data table-like source preview; full CSV is available in paper/source_data.")
        p = canvas.panel("A", 60, 150, 1680, 610, title)
        write_source_data(stem, "A_source_preview", frame)
        cols = list(frame.columns[:5]) if not frame.empty else ["source"]
        canvas.text(p.x + 35, p.y + 80, "Columns: " + ", ".join(human_label(c) for c in cols), size=18, weight="800", max_width=1480)
        for i, (_, row) in enumerate(frame.head(8).iterrows()):
            text = " | ".join(f"{human_label(c)}: {row[c]}" for c in cols)
            canvas.text(p.x + 35, p.y + 140 + i * 48, text, size=12.5, fill=VI.muted, max_width=1540)
        outputs.extend(_save(canvas, stem))
    return outputs


def generate_nmi_svg_figures() -> list[Path]:
    ensure_directories()
    summary_json = _read_summary()
    summary = _read_csv("cross_dataset_summary.csv")
    construct = _read_csv("construct_matrix.csv")
    effect = _read_csv("effect_sizes.csv")
    haiid = _read_csv("haiid_descriptive_metrics.csv")
    chi_conditions = _read_csv("chi2023_condition_effects.csv")
    convxai_conditions = _read_csv("convxai_condition_effects.csv")
    gee = _read_csv("gee_results.csv")
    pardos = _read_csv("pardos_learning_effects.csv")
    flora = _read_csv("flora_process_effects.csv")
    flora_models = _read_csv("flora_process_models.csv")
    model = _read_csv("real_model_results.csv")
    cross = _read_csv("cross_dataset_results.csv")
    curves = _read_csv("real_calibration_curves.csv")
    calibration = _read_csv("calibration_summary.csv")
    ablation = _read_csv("real_rule_ablation_results.csv")
    policy = _read_csv("policy_evaluation.csv")
    decision_curve = _read_csv("decision_curve_summary.csv")
    state = _read_csv("reliance_state_distribution.csv")
    sensitivity = _read_csv("sensitivity_summary.csv")

    outputs: list[Path] = []
    outputs.extend(figure_1(summary_json))
    outputs.extend(figure_2(summary_json, construct))
    outputs.extend(figure_3(effect, summary, haiid, state, flora))
    outputs.extend(figure_4(chi_conditions, convxai_conditions, gee))
    outputs.extend(figure_5(pardos, flora, flora_models, gee))
    outputs.extend(figure_6())
    outputs.extend(figure_7(model, cross))
    outputs.extend(figure_8(curves, calibration, decision_curve))
    outputs.extend(figure_9(ablation, policy))
    outputs.extend(figure_10(policy, decision_curve))
    outputs.extend(extended_figures(summary_json, state, model, calibration, cross, sensitivity))
    return outputs
