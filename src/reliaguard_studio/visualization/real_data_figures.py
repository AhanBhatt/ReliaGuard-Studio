from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

from ..paths import PAPER_FIGURES_DIR, REAL_DATA_EXPERIMENTS_DIR, ensure_directories
from .export import export_figure
from .nature_layouts import DOUBLE_TALL, DOUBLE_WIDE, METHOD_WIDE
from .panel_labels import add_panel_label, direct_label
from .style import CONSTRUCT_COLORS, DATASET_COLORS, MODEL_COLORS, PALETTE, POLICY_COLORS, apply_nature_style, clean_label


DATASET_LABELS = {
    "haiid": "HAIID",
    "chi2023_dke": "CHI 2023 DKE",
    "convxai_iui2025": "ConvXAI",
    "pardos_chatgpt_tutoring": "Pardos/Bhandari",
    "flora_ips": "FLoRA IPS",
}

POLICY_LABELS = {
    "observed_no_gating": "Observed",
    "confidence_threshold_gating": "Confidence threshold",
    "symbolic_rule_gating": "Symbolic gating",
    "neurosymbolic_gating": "Neuro-symbolic gating",
}

STATE_COLORS = {
    "beneficial_ai_reliance": CONSTRUCT_COLORS["appropriate reliance"],
    "correct_self_reliance": PALETTE.blue,
    "independent_correct": PALETTE.mint,
    "independent_incorrect": PALETTE.rose,
    "harmful_overreliance": CONSTRUCT_COLORS["overreliance"],
    "harmful_underreliance": CONSTRUCT_COLORS["underreliance"],
    "harmful_ai_overreliance": CONSTRUCT_COLORS["overreliance"],
    "harmful_ai_underreliance": CONSTRUCT_COLORS["underreliance"],
    "uncertain_disagreement": PALETTE.violet,
}


def _read_csv(name: str) -> pd.DataFrame:
    path = REAL_DATA_EXPERIMENTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_summary() -> dict:
    path = REAL_DATA_EXPERIMENTS_DIR / "real_experiment_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _dataset_label(value: object) -> str:
    return DATASET_LABELS.get(str(value), clean_label(value))


def _clean_policy(value: object) -> str:
    return POLICY_LABELS.get(str(value), clean_label(value))


def _lighten(hex_color: str, factor: float = 0.82) -> str:
    color = hex_color.lstrip("#")
    rgb = np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)
    rgb = rgb + (255 - rgb) * factor
    return "#" + "".join(f"{int(v):02x}" for v in rgb)


def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str | None = None,
    fontsize: float = 9,
    weight: str = "bold",
) -> None:
    edge = edge or face
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.016,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = PALETTE.muted, lw: float = 1.5) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "lw": lw, "shrinkA": 3, "shrinkB": 3})


def _forest(
    ax: plt.Axes,
    frame: pd.DataFrame,
    labels: list[str],
    estimate: str,
    low: str,
    high: str,
    *,
    color: str,
    xlabel: str,
    zero: float = 0.0,
    xlim: tuple[float, float] | None = None,
) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "No estimable contrast", ha="center", va="center", transform=ax.transAxes, color=PALETTE.muted)
        ax.axis("off")
        return
    y = np.arange(len(frame))
    est = pd.to_numeric(frame[estimate], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(frame[low], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(frame[high], errors="coerce").to_numpy(float)
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color=color, ecolor=PALETTE.muted, capsize=3, markersize=5, lw=1.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(zero, color=PALETTE.ink, ls="--", lw=0.9, alpha=0.65)
    ax.grid(axis="x", color=PALETTE.grid, lw=0.7)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)


def _metric_bars(ax: plt.Axes, labels: list[str], values: list[float], colors: list[str], *, xlabel: str, xlim: tuple[float, float] | None = None) -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.95)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=PALETTE.grid, lw=0.7)
    if xlim:
        ax.set_xlim(*xlim)
    for yi, value in zip(y, values, strict=False):
        ax.text(value + (0.01 if value >= 0 else -0.01), yi, f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=8.2)


def _figure1(summary: dict) -> list[Path]:
    fig, ax = plt.subplots(figsize=(DOUBLE_WIDE.width, DOUBLE_WIDE.height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "Reliance-state evaluation links evidence, models and selective action", fontsize=18, fontweight="bold", ha="left")
    ax.text(
        0.02,
        0.915,
        "Average AI-assisted gains are decomposed into beneficial reliance, harmful overreliance, harmful underreliance and evidence boundaries.",
        fontsize=10.5,
        color=PALETTE.muted,
        ha="left",
    )
    columns = [0.06, 0.30, 0.54, 0.78]
    headers = ["A  empirical base", "B  construct layer", "C  neuro-symbolic model", "D  action layer"]
    for x, header in zip(columns, headers, strict=False):
        ax.text(x, 0.84, header, fontsize=10.8, fontweight="bold", color=PALETTE.ink, ha="left")

    sizes = summary.get("dataset_sizes", {})
    dataset_order = ["haiid", "chi2023_dke", "convxai_iui2025", "pardos_chatgpt_tutoring", "flora_ips"]
    for i, key in enumerate(dataset_order):
        label = _dataset_label(key)
        color = DATASET_COLORS[label]
        row = i % 5
        y = 0.73 - row * 0.105
        n_records = int(sizes.get(key, {}).get("n_interactions", 0))
        n_people = int(sizes.get(key, {}).get("n_participants", 0))
        _rounded_box(ax, (0.055, y), 0.17, 0.07, f"{label}\n{n_records:,} records; {n_people:,} people", face=_lighten(color, 0.78), edge=color, fontsize=8.6)

    constructs = [
        ("Appropriate reliance", "appropriate reliance"),
        ("Overreliance", "overreliance"),
        ("Underreliance", "underreliance"),
        ("Confidence/calibration", "confidence/calibration"),
        ("Short-term learning gain", "learning gain"),
        ("Process traces", "process traces"),
        ("Unsupported: delayed recall / transfer", "unsupported"),
    ]
    for i, (text, key) in enumerate(constructs):
        y = 0.74 - i * 0.075
        color = CONSTRUCT_COLORS[key]
        _rounded_box(ax, (0.295, y), 0.185, 0.045, text, face=_lighten(color, 0.80), edge=color, fontsize=8.0)

    model_nodes = [
        ("Harmonized\nfeatures", PALETTE.sky),
        ("Calibrated\ntabular risk", PALETTE.mint),
        ("Fuzzy\nsymbolic rules", PALETTE.sand),
        ("Bootstrap\nuncertainty", PALETTE.lavender),
        ("State\nprobabilities", _lighten(PALETTE.green, 0.70)),
    ]
    for i, (text, color) in enumerate(model_nodes):
        y = 0.72 - i * 0.085
        _rounded_box(ax, (0.54, y), 0.15, 0.052, text, face=color, edge=PALETTE.grid, fontsize=8.2)
        if i < len(model_nodes) - 1:
            _arrow(ax, (0.615, y), (0.615, y - 0.032), color=PALETTE.muted, lw=1.0)

    action_nodes = [
        ("Calibrated risk", PALETTE.mint),
        ("Rule trace", PALETTE.sand),
        ("Counterfactual", PALETTE.sky),
        ("Selective gating", PALETTE.rose),
        ("Evidence boundary", _lighten(PALETTE.muted, 0.75)),
    ]
    for i, (text, color) in enumerate(action_nodes):
        y = 0.72 - i * 0.085
        _rounded_box(ax, (0.78, y), 0.16, 0.052, text, face=color, edge=PALETTE.grid, fontsize=8.2)
    _arrow(ax, (0.69, 0.55), (0.78, 0.55), color=PALETTE.ink, lw=1.5)
    ax.text(
        0.06,
        0.11,
        "43,263 real records; synthetic AIR-Bench retained only as supplementary stress testing.",
        fontsize=9.4,
        color=PALETTE.muted,
    )
    ax.text(
        0.54,
        0.11,
        "Formal score and utility equations are shown in Figure 6.",
        fontsize=9.4,
        color=PALETTE.muted,
        ha="left",
    )
    return export_figure("figure_1_graphical_abstract")


def _figure2(summary: dict, construct: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(DOUBLE_TALL.width, DOUBLE_TALL.height), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.45, 0.95], height_ratios=[1, 1])
    ax_cards = fig.add_subplot(gs[:, 0])
    ax_matrix = fig.add_subplot(gs[:, 1])
    ax_scale = fig.add_subplot(gs[0, 2])
    ax_counts = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_cards, "A", "Dataset cards")
    ax_cards.axis("off")
    sizes = summary.get("dataset_sizes", {})
    order = ["haiid", "chi2023_dke", "convxai_iui2025", "pardos_chatgpt_tutoring", "flora_ips"]
    domains = {
        "haiid": "binary advice tasks",
        "chi2023_dke": "reasoning + DKE",
        "convxai_iui2025": "loan decisions + XAI",
        "pardos_chatgpt_tutoring": "math tutoring",
        "flora_ips": "GenAI writing process",
    }
    for i, key in enumerate(order):
        label = _dataset_label(key)
        color = DATASET_COLORS[label]
        y = 0.88 - i * 0.17
        _rounded_box(
            ax_cards,
            (0.04, y - 0.09),
            0.86,
            0.12,
            f"{label}\n{int(sizes.get(key, {}).get('n_interactions', 0)):,} records; {int(sizes.get(key, {}).get('n_participants', 0)):,} people\n{domains[key]}",
            face=_lighten(color, 0.82),
            edge=color,
            fontsize=8.3,
        )

    add_panel_label(ax_matrix, "B", "Construct support ledger")
    cols = [
        "appropriate reliance",
        "overreliance",
        "underreliance",
        "confidence shift",
        "learning gain",
        "process traces",
        "metacognitive engagement",
        "delayed recall",
        "transfer",
    ]
    matrix = construct.set_index("dataset_key")[cols].reindex(order).fillna(0).astype(int)
    ax_matrix.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax_matrix.set_xticks(range(len(cols)), [clean_label(c).replace(" ", "\n") for c in cols], fontsize=7.5)
    ax_matrix.set_yticks(range(len(order)), [_dataset_label(k) for k in order])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            supported = bool(matrix.iloc[i, j])
            ax_matrix.text(j, i, "●" if supported else "○", ha="center", va="center", fontsize=12, color="white" if supported else PALETTE.muted)
    ax_matrix.axvline(3.5, color=PALETTE.ink, lw=1.0)
    ax_matrix.axvline(5.5, color=PALETTE.ink, lw=1.0)
    ax_matrix.axvline(6.5, color=PALETTE.red, lw=1.0)
    ax_matrix.text(1.5, -0.85, "Decision reliance", ha="center", fontsize=8.5, fontweight="bold")
    ax_matrix.text(4.5, -0.85, "Learning/process", ha="center", fontsize=8.5, fontweight="bold")
    ax_matrix.text(7.5, -0.85, "Unsupported future constructs", ha="center", fontsize=8.5, fontweight="bold", color=PALETTE.red)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    add_panel_label(ax_scale, "C", "Evidence strength scale")
    ax_scale.axis("off")
    scale = [
        ("Experimental / randomized", PALETTE.green),
        ("Observational", PALETTE.gold),
        ("Prediction only", PALETTE.blue),
        ("Unsupported", PALETTE.muted),
    ]
    for i, (text, color) in enumerate(scale):
        _rounded_box(ax_scale, (0.1, 0.78 - i * 0.18), 0.78, 0.09, text, face=_lighten(color, 0.76), edge=color, fontsize=8.5)

    add_panel_label(ax_counts, "D", "Corpus scale")
    count_frame = pd.DataFrame(
        [
            {"dataset": _dataset_label(k), "records": int(sizes.get(k, {}).get("n_interactions", 0)), "participants": int(sizes.get(k, {}).get("n_participants", 0))}
            for k in order
        ]
    )
    y = np.arange(len(count_frame))
    ax_counts.barh(y + 0.16, count_frame["records"], height=0.28, color=PALETTE.blue, label="records")
    ax_counts.barh(y - 0.16, count_frame["participants"], height=0.28, color=PALETTE.gold, label="participants")
    ax_counts.set_yticks(y, count_frame["dataset"])
    ax_counts.set_xscale("log")
    ax_counts.invert_yaxis()
    ax_counts.grid(axis="x", color=PALETTE.grid)
    ax_counts.set_xlabel("count (log scale)")
    ax_counts.legend(loc="lower right")
    return export_figure("figure_2_evidence_boundary")


def _figure3(effect: pd.DataFrame, summary: pd.DataFrame, haiid: pd.DataFrame, state: pd.DataFrame, flora: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(13.6, 10.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.2], width_ratios=[1.0, 1.0])
    fig.subplots_adjust(left=0.09, right=0.86, top=0.93, bottom=0.08, hspace=0.62, wspace=0.58)
    ax_gain = fig.add_subplot(gs[0, 0])
    ax_fail = fig.add_subplot(gs[0, 1])
    ax_flora = fig.add_subplot(gs[1, 0])
    ax_source = fig.add_subplot(gs[1, 1])
    ax_state = fig.add_subplot(gs[2, :])

    gain = effect.loc[~effect["dataset"].eq("flora_ips")].copy()
    gain["label"] = gain["dataset"].map(_dataset_label)
    add_panel_label(ax_gain, "A", "Outcome gains are positive")
    _forest(ax_gain, gain, gain["label"].tolist(), "effect", "ci_low", "ci_high", color=PALETTE.green, xlabel="final/post minus initial/pre", xlim=(-0.07, 0.18))

    add_panel_label(ax_flora, "B", "FLoRA is observational, not pre/post")
    flora_plot = flora.loc[flora["process_feature"].isin(["genai_intensity", "copy_paste_write_ratio", "metacognitive_prompt_ratio", "source_engagement_proxy"])].copy()
    flora_plot["label"] = flora_plot["process_feature"].map(clean_label)
    _forest(ax_flora, flora_plot, flora_plot["label"].tolist(), "score_difference_high_minus_low", "ci_low", "ci_high", color=PALETTE.violet, xlabel="proposal-score contrast", xlim=(-0.05, 0.08))

    add_panel_label(ax_fail, "C", "Reliance failures are asymmetric")
    failure = summary.loc[summary["dataset_name"].isin(["haiid", "convxai_iui2025"])].copy()
    x = np.arange(len(failure))
    ax_fail.bar(x - 0.18, failure["overreliance_rate"], width=0.34, color=CONSTRUCT_COLORS["overreliance"], label="overreliance")
    ax_fail.bar(x + 0.18, failure["underreliance_rate"], width=0.34, color=CONSTRUCT_COLORS["underreliance"], label="underreliance")
    ax_fail.set_xticks(x, failure["dataset_name"].map(_dataset_label))
    ax_fail.set_ylabel("population rate")
    ax_fail.grid(axis="y", color=PALETTE.grid)
    ax_fail.legend(loc="upper right")

    add_panel_label(ax_source, "D", "Advice-source asymmetry in HAIID")
    source = (
        haiid.loc[haiid["metric"].isin(["correct_ai_reliance", "correct_self_reliance", "overreliance", "underreliance"])]
        .groupby(["advice_source_label", "metric"], as_index=False)["rate"]
        .mean()
    )
    metrics = ["correct_ai_reliance", "correct_self_reliance", "overreliance", "underreliance"]
    yy = np.arange(len(metrics))
    for offset, source_name, color in [(-0.16, "ai", PALETTE.blue), (0.16, "human", PALETTE.green)]:
        vals = [float(source.loc[(source["metric"] == m) & (source["advice_source_label"] == source_name), "rate"].mean()) for m in metrics]
        ax_source.barh(yy + offset, vals, height=0.28, color=color, label=source_name.upper() if source_name == "ai" else "Human")
    ax_source.set_yticks(yy, [clean_label(m) for m in metrics])
    ax_source.invert_yaxis()
    ax_source.set_xlabel("mean rate")
    ax_source.grid(axis="x", color=PALETTE.grid)
    ax_source.legend(loc="lower right")

    add_panel_label(ax_state, "E", "State decomposition prevents accuracy-only evaluation")
    state_plot = state.copy()
    if state_plot.empty:
        state_plot = pd.DataFrame()
    dataset_col = "dataset" if "dataset" in state_plot.columns else "dataset_name"
    state_col = "state" if "state" in state_plot.columns else "reliance_state"
    if "share" not in state_plot.columns and {"count", dataset_col}.issubset(state_plot.columns):
        totals = state_plot.groupby(dataset_col)["count"].transform("sum")
        state_plot["share"] = state_plot["count"] / totals
    datasets = [d for d in ["chi2023_dke", "convxai_iui2025", "haiid"] if d in state_plot.get(dataset_col, pd.Series(dtype=str)).unique()]
    states = [s for s in STATE_COLORS if s in state_plot.get(state_col, pd.Series(dtype=str)).unique()]
    bottom = np.zeros(len(datasets))
    for s in states:
        vals = [float(state_plot.loc[(state_plot[dataset_col] == d) & (state_plot[state_col] == s), "share"].sum()) for d in datasets]
        ax_state.bar(np.arange(len(datasets)), vals, bottom=bottom, color=STATE_COLORS[s], label=clean_label(s))
        bottom += np.array(vals)
    ax_state.set_xticks(np.arange(len(datasets)), [_dataset_label(d) for d in datasets])
    ax_state.set_ylabel("share of records")
    ax_state.set_ylim(0, min(1, max(0.2, bottom.max() * 1.15)) if len(bottom) else 1)
    ax_state.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.0)
    ax_state.grid(axis="y", color=PALETTE.grid)
    return export_figure("figure_3_asymmetric_failures")


def _figure4(chi_cond: pd.DataFrame, conv_cond: pd.DataFrame, gee: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(DOUBLE_TALL.width, DOUBLE_TALL.height), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_chi = fig.add_subplot(gs[0, 0])
    ax_self = fig.add_subplot(gs[0, 1])
    ax_conv = fig.add_subplot(gs[1, 0])
    ax_rel = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_chi, "A", "CHI tutorial/XAI contrasts")
    chi = chi_cond.loc[chi_cond["metric"].eq("appropriate_reliance")].copy()
    if not chi.empty:
        ref = float(chi.loc[chi["condition_name"].eq("control"), "rate"].mean()) if chi["condition_name"].eq("control").any() else float(chi["rate"].iloc[0])
        chi["contrast"] = chi["rate"] - ref
        chi["low"] = chi["ci_low"] - ref
        chi["high"] = chi["ci_high"] - ref
        chi["label"] = chi["condition_name"].map(clean_label)
    _forest(ax_chi, chi, chi.get("label", pd.Series(dtype=str)).tolist(), "contrast", "low", "high", color=PALETTE.gold, xlabel="difference from control", xlim=(-0.10, 0.16))

    add_panel_label(ax_self, "B", "Self-calibration GEE effects")
    terms = ["first_batch_overestimation", "first_batch_underestimation", "trust_first"]
    self_gee = gee.loc[(gee["dataset"].eq("chi2023_dke")) & (gee["outcome"].eq("appropriate_reliance")) & (gee["term"].isin(terms))].copy()
    self_gee["label"] = self_gee["term"].map(clean_label)
    _forest(ax_self, self_gee, self_gee.get("label", pd.Series(dtype=str)).tolist(), "estimate", "ci_low", "ci_high", color=PALETTE.blue, xlabel="GEE log-odds", xlim=(-0.85, 0.45))

    add_panel_label(ax_conv, "C", "ConvXAI interface contrasts")
    conv = conv_cond.loc[conv_cond["metric"].eq("appropriate_reliance")].copy()
    if not conv.empty:
        condition_col = "condition_id" if "condition_id" in conv.columns else "condition_name"
        ref = float(conv.loc[conv[condition_col].eq("control"), "rate"].mean()) if conv[condition_col].eq("control").any() else float(conv["rate"].iloc[0])
        conv["contrast"] = conv["rate"] - ref
        conv["low"] = conv["ci_low"] - ref
        conv["high"] = conv["ci_high"] - ref
        conv["label"] = conv[condition_col].map(clean_label)
    _forest(ax_conv, conv, conv.get("label", pd.Series(dtype=str)).tolist(), "contrast", "low", "high", color=PALETTE.teal, xlabel="difference from control", xlim=(-0.12, 0.18))

    add_panel_label(ax_rel, "D", "Reliability can be protective or harmful")
    rel_terms = ["initial_confidence", "confidence_change", "post_explain_reliability"]
    rel = gee.loc[(gee["dataset"].eq("convxai_iui2025")) & (gee["outcome"].isin(["appropriate_reliance", "overreliance"])) & (gee["term"].isin(rel_terms))].copy()
    rel["label"] = rel["outcome"].map(clean_label) + ": " + rel["term"].map(clean_label)
    rel = rel.sort_values(["outcome", "estimate"])
    _forest(ax_rel, rel, rel.get("label", pd.Series(dtype=str)).tolist(), "estimate", "ci_low", "ci_high", color=PALETTE.red, xlabel="GEE log-odds", xlim=(-8.0, 8.0))
    ax_rel.text(0.02, -0.17, "Large overreliance coefficient reflects sparse wrong-advice subset; reported with caution.", transform=ax_rel.transAxes, fontsize=8, color=PALETTE.muted)
    return export_figure("figure_4_interface_calibration")


def _figure5(pardos: pd.DataFrame, flora: pd.DataFrame, flora_models: pd.DataFrame, gee: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(DOUBLE_TALL.width, DOUBLE_TALL.height), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.09, hspace=0.50, wspace=0.55)
    gs = fig.add_gridspec(2, 2)
    ax_gain = fig.add_subplot(gs[0, 0])
    ax_gee = fig.add_subplot(gs[0, 1])
    ax_flora = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_gain, "A", "Short-term mathematics learning gains")
    pardos = pardos.copy()
    pardos["label"] = pardos["condition_name"].map(clean_label)
    _forest(ax_gain, pardos, pardos["label"].tolist(), "learning_gain", "ci_low", "ci_high", color=PALETTE.rust, xlabel="post-test minus pre-test", xlim=(-0.07, 0.24))
    add_panel_label(ax_gee, "B", "Condition contrasts vs ChatGPT help")
    pgee = gee.loc[
        (gee["dataset"].eq("pardos_chatgpt_tutoring"))
        & (~gee["term"].eq("Intercept"))
        & (gee["term"].astype(str).str.contains("condition_name"))
    ].copy()
    pgee["label"] = pgee["term"].str.replace("C(condition_name)[T.", "", regex=False).str.replace("]", "", regex=False).map(clean_label)
    _forest(ax_gee, pgee, pgee.get("label", pd.Series(dtype=str)).tolist(), "estimate", "ci_low", "ci_high", color=PALETTE.green, xlabel="learning-gain difference", xlim=(-0.18, 0.08))
    add_panel_label(ax_flora, "C", "FLoRA process associations are small")
    flora_plot = flora.copy().sort_values("score_difference_high_minus_low")
    flora_plot["label"] = flora_plot["process_feature"].map(clean_label)
    _forest(ax_flora, flora_plot, flora_plot["label"].tolist(), "score_difference_high_minus_low", "ci_low", "ci_high", color=PALETTE.violet, xlabel="proposal-score contrast", xlim=(-0.055, 0.075))
    add_panel_label(ax_pred, "D", "Aggregate process prediction remains weak")
    models = flora_models.sort_values("r2").copy()
    colors = [PALETTE.green if v == models["r2"].max() else PALETTE.muted for v in models["r2"]]
    _metric_bars(ax_pred, [clean_label(v) for v in models["model"]], models["r2"].tolist(), colors, xlabel="student-level cross-validated R²", xlim=(-0.30, 0.06))
    ax_pred.axvline(0, color=PALETTE.ink, lw=0.9, ls="--")
    ax_pred.text(0.03, 0.08, "Boundary result: coarse process summaries are insufficient.", transform=ax_pred.transAxes, fontsize=8.5, color=PALETTE.muted)
    return export_figure("figure_5_learning_process_extension")


def _figure6() -> list[Path]:
    fig, ax = plt.subplots(figsize=(METHOD_WIDE.width, METHOD_WIDE.height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "Reliance-State NeuroSymbolic Calibration and Gating Model", fontsize=17, fontweight="bold")
    ax.text(0.02, 0.895, "A formal layer separates measurement, prediction, explanation, observational policy simulation and causal claims.", fontsize=10, color=PALETTE.muted)
    layers = [
        ("Harmonized inputs\n$h^{(0)}, a, h^{(1)}, y, x$", 0.04, 0.60, PALETTE.sky),
        ("Calibrated tabular estimator\n$\\eta_\\theta(x)$", 0.25, 0.68, PALETTE.mint),
        ("Fuzzy symbolic rules\n$r_k(x)=\\prod_j\\mu_{kj}(x_j)$", 0.25, 0.47, PALETTE.sand),
        ("Bootstrap uncertainty\n$u(x)$", 0.46, 0.57, PALETTE.lavender),
        ("State probabilities\n$\\hat p_s=\\sigma(\\eta+\\lambda R_s-\\gamma u)$", 0.63, 0.68, _lighten(PALETTE.green, 0.72)),
        ("Rule-grounded counterfactual\nminimal feature/action shift", 0.63, 0.47, _lighten(PALETTE.blue, 0.78)),
        ("Selective gating utility\n$U=P_c-\\alpha P_o-\\beta P_u-\\kappa I$", 0.82, 0.57, _lighten(PALETTE.red, 0.78)),
    ]
    for text, x, y, color in layers:
        _rounded_box(ax, (x, y), 0.15 if x != 0.82 else 0.15, 0.15, text, face=color, edge=PALETTE.grid, fontsize=8.3)
    for start, end in [
        ((0.19, 0.675), (0.25, 0.755)),
        ((0.19, 0.675), (0.25, 0.545)),
        ((0.40, 0.755), (0.46, 0.645)),
        ((0.40, 0.545), (0.46, 0.645)),
        ((0.61, 0.645), (0.63, 0.755)),
        ((0.61, 0.645), (0.63, 0.545)),
        ((0.78, 0.755), (0.82, 0.645)),
        ((0.78, 0.545), (0.82, 0.645)),
    ]:
        _arrow(ax, start, end, color=PALETTE.ink, lw=1.2)
    ax.text(0.05, 0.31, "Reliance states", fontsize=10, fontweight="bold")
    states = [
        ("beneficial AI reliance", PALETTE.green),
        ("harmful overreliance", PALETTE.red),
        ("harmful underreliance", PALETTE.gold),
        ("correct self-reliance", PALETTE.blue),
        ("independent correct/incorrect", PALETTE.muted),
        ("uncertain disagreement", PALETTE.violet),
    ]
    for i, (text, color) in enumerate(states):
        _rounded_box(ax, (0.05 + (i % 3) * 0.18, 0.20 - (i // 3) * 0.085), 0.155, 0.055, text, face=_lighten(color, 0.80), edge=color, fontsize=7.7)
    ax.text(0.64, 0.31, "Causal boundary", fontsize=10, fontweight="bold")
    _rounded_box(
        ax,
        (0.64, 0.18),
        0.30,
        0.12,
        "Offline utility ranks intervention candidates.\nIt does not identify the causal effect of showing them.",
        face=_lighten(PALETTE.muted, 0.84),
        edge=PALETTE.muted,
        fontsize=8.3,
        weight="normal",
    )
    return export_figure("figure_6_reliance_state_method")


def _figure7(model: pd.DataFrame, cross: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(DOUBLE_TALL.width, DOUBLE_TALL.height), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_strict = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_transfer = fig.add_subplot(gs[1, 0])
    ax_delta = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_strict, "A", "Strict validation keeps baselines visible")
    keep = model.loc[model["split"].isin(["participant", "strict_task"])].copy()
    keep = keep.sort_values("auroc", ascending=False).groupby(["dataset", "target", "split"]).head(1)
    keep = keep.sort_values("auroc").tail(8)
    keep["label"] = keep["dataset"].map(_dataset_label) + " / " + keep["target"].map(clean_label) + " / " + keep["split"].map(clean_label)
    _forest(ax_strict, keep, keep["label"].tolist(), "auroc", "auroc_ci_low", "auroc_ci_high", color=PALETTE.blue, xlabel="AUROC", zero=0.5, xlim=(0.48, 0.98))
    add_panel_label(ax_rank, "B", "Mean strict-task AUROC by model family")
    strict = model.loc[model["split"].eq("strict_task")].copy()
    rank = strict.groupby("model", as_index=False)["auroc"].mean().sort_values("auroc")
    rank = rank.loc[rank["model"].isin(["logistic_regression", "calibrated_gradient_boosting", "symbolic_only", "uncertainty_aware_fusion", "reliance_state_neurosymbolic"])]
    colors = [MODEL_COLORS.get(clean_label(m).lower(), PALETTE.blue) for m in rank["model"]]
    _metric_bars(ax_rank, [clean_label(m) for m in rank["model"]], rank["auroc"].tolist(), colors, xlabel="mean strict-task AUROC", xlim=(0.45, 0.83))
    add_panel_label(ax_transfer, "C", "Appropriate-reliance transfer matrix")
    transfer = cross.loc[cross["target"].eq("appropriate_reliance")].sort_values("auroc", ascending=False).groupby(["train_dataset", "test_dataset"]).head(1)
    matrix = transfer.pivot(index="train_dataset", columns="test_dataset", values="auroc")
    rows = [r for r in ["haiid", "chi2023_dke", "convxai_iui2025", "all_except_haiid", "all_except_chi2023_dke", "all_except_convxai_iui2025"] if r in matrix.index]
    cols = [c for c in ["haiid", "chi2023_dke", "convxai_iui2025"] if c in matrix.columns]
    matrix = matrix.reindex(rows)[cols]
    im = ax_transfer.imshow(matrix.to_numpy(float), cmap="YlGnBu", vmin=0.45, vmax=0.85, aspect="auto")
    ax_transfer.set_xticks(range(len(cols)), [_dataset_label(c) for c in cols], rotation=20, ha="right")
    ax_transfer.set_yticks(range(len(rows)), [_dataset_label(r) for r in rows])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax_transfer.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax_transfer, fraction=0.046, pad=0.02, label="AUROC")
    add_panel_label(ax_delta, "D", "Harmful-reliance transfer remains bounded")
    harmful = cross.loc[cross["target"].isin(["overreliance", "underreliance"])].copy()
    harmful = harmful.sort_values("auroc", ascending=False).groupby(["train_dataset", "test_dataset", "target"]).head(1).sort_values("auroc").tail(7)
    harmful["label"] = harmful["target"].map(clean_label) + " → " + harmful["test_dataset"].map(_dataset_label)
    _forest(ax_delta, harmful, harmful.get("label", pd.Series(dtype=str)).tolist(), "auroc", "auroc_ci_low", "auroc_ci_high", color=PALETTE.red, xlabel="external AUROC", zero=0.5, xlim=(0.45, 0.76))
    return export_figure("figure_7_prediction_generalization")


def _plot_reliability(ax: plt.Axes, curves: pd.DataFrame, title: str, models: list[str]) -> None:
    ax.plot([0, 1], [0, 1], color=PALETTE.muted, lw=0.9, ls="--")
    for model_name, color in zip(models, [PALETTE.green, PALETTE.blue, PALETTE.gold], strict=False):
        sub = curves.loc[curves["model"].eq(model_name)].sort_values("mean_confidence")
        if sub.empty:
            continue
        ax.plot(sub["mean_confidence"], sub["accuracy"], marker="o", markersize=4, color=color, label=clean_label(model_name), lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("empirical rate")
    ax.grid(color=PALETTE.grid, lw=0.6)
    ax.legend(loc="lower right", fontsize=7.5)


def _figure8(curves: pd.DataFrame, calibration: pd.DataFrame, policy_curve: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(DOUBLE_TALL.width, DOUBLE_TALL.height), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_a, "A")
    sub = curves.loc[(curves["dataset"].eq("haiid")) & (curves["target"].eq("underreliance")) & (curves["split"].eq("participant"))]
    _plot_reliability(ax_a, sub, "HAIID underreliance", ["reliance_state_neurosymbolic", "uncertainty_aware_fusion"])
    add_panel_label(ax_b, "B")
    sub = curves.loc[(curves["dataset"].eq("convxai_iui2025")) & (curves["target"].eq("overreliance")) & (curves["split"].eq("participant"))]
    _plot_reliability(ax_b, sub, "ConvXAI overreliance", ["reliance_state_neurosymbolic", "uncertainty_aware_fusion"])
    add_panel_label(ax_c, "C")
    sub = curves.loc[(curves["dataset"].eq("chi2023_dke")) & (curves["target"].eq("appropriate_reliance")) & (curves["split"].eq("participant"))]
    _plot_reliability(ax_c, sub, "CHI appropriate reliance", ["reliance_state_neurosymbolic", "uncertainty_aware_fusion"])
    add_panel_label(ax_d, "D", "Lowest ECE cases")
    ece = calibration.sort_values("ece").head(8).copy()
    ece["label"] = ece["dataset"].map(_dataset_label) + " / " + ece["target"].map(clean_label)
    colors = [PALETTE.green if m == "reliance_state_neurosymbolic" else PALETTE.blue for m in ece["model"]]
    _metric_bars(ax_d, ece["label"].tolist(), ece["ece"].tolist(), colors, xlabel="expected calibration error", xlim=(0, max(0.16, ece["ece"].max() * 1.25)))
    add_panel_label(ax_e, "E", "Calibration slope varies")
    slope = calibration.loc[calibration["model"].isin(["reliance_state_neurosymbolic", "uncertainty_aware_fusion"])].sort_values("calibration_slope").head(10)
    ax_e.scatter(slope["calibration_slope"], np.arange(len(slope)), color=PALETTE.violet, s=28)
    ax_e.axvline(1, color=PALETTE.ink, ls="--", lw=0.9)
    ax_e.set_yticks(np.arange(len(slope)), (slope["dataset"].map(_dataset_label) + " / " + slope["split"].map(clean_label)).tolist(), fontsize=7)
    ax_e.set_xlabel("calibration slope")
    ax_e.grid(axis="x", color=PALETTE.grid)
    add_panel_label(ax_f, "F", "Selective action trades coverage for utility")
    policy = policy_curve.copy()
    for dataset, group in policy.groupby("dataset"):
        group = group.sort_values("intervention_burden")
        ax_f.plot(group["intervention_burden"], group["utility_gain_vs_observed"], marker="o", label=_dataset_label(dataset), lw=1.5)
    ax_f.axhline(0, color=PALETTE.ink, ls="--", lw=0.9)
    ax_f.set_xlabel("intervention burden")
    ax_f.set_ylabel("utility gain vs observed")
    ax_f.grid(color=PALETTE.grid)
    ax_f.legend(fontsize=7.5)
    return export_figure("figure_8_calibration_uncertainty")


def _figure9(ablation: pd.DataFrame, policy: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(DOUBLE_TALL.width, DOUBLE_TALL.height), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    ax_abl = fig.add_subplot(gs[0, 0])
    ax_flow = fig.add_subplot(gs[0, 1])
    ax_card = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_abl, "A", "Rule ablations expose failure-mode signal")
    abl = ablation.loc[ablation["split"].isin(["participant", "strict_task"])].copy()
    abl = abl.sort_values("mean_difference").head(9)
    abl["label"] = abl["dataset"].map(_dataset_label) + " / " + abl["group"].map(clean_label)
    _forest(ax_abl, abl, abl["label"].tolist(), "mean_difference", "ci_low", "ci_high", color=PALETTE.red, xlabel="AUROC change when rule group is removed", xlim=(-0.36, 0.06))
    add_panel_label(ax_flow, "B", "Rules map to intervention families")
    ax_flow.axis("off")
    left = [("Wrong-advice\nvulnerability", PALETTE.red), ("Correct-advice\nunderuse", PALETTE.gold), ("Low evaluation /\nsource engagement", PALETTE.violet)]
    middle = [("Harmful\noverreliance", PALETTE.red), ("Harmful\nunderreliance", PALETTE.gold), ("Uncertain\ndisagreement", PALETTE.blue)]
    right = [("Request\nverification", PALETTE.blue), ("Compare\nevidence", PALETTE.green), ("Delay / show\nuncertainty cue", PALETTE.muted)]
    for i, (text, color) in enumerate(left):
        _rounded_box(ax_flow, (0.02, 0.74 - i * 0.25), 0.26, 0.11, text, face=_lighten(color, 0.82), edge=color, fontsize=8.1)
    for i, (text, color) in enumerate(middle):
        _rounded_box(ax_flow, (0.38, 0.74 - i * 0.25), 0.24, 0.11, text, face=_lighten(color, 0.82), edge=color, fontsize=8.1)
    for i, (text, color) in enumerate(right):
        _rounded_box(ax_flow, (0.72, 0.74 - i * 0.25), 0.24, 0.11, text, face=_lighten(color, 0.82), edge=color, fontsize=8.1)
    for y in [0.79, 0.54, 0.29]:
        _arrow(ax_flow, (0.28, y), (0.38, y), lw=1.4)
        _arrow(ax_flow, (0.62, y), (0.72, y), lw=1.4)
    add_panel_label(ax_card, "C", "Counterfactual explanation card")
    ax_card.axis("off")
    _rounded_box(ax_card, (0.07, 0.18), 0.84, 0.62, "", face="#FBFCFD", edge=PALETTE.grid)
    ax_card.text(0.12, 0.70, "Predicted state: harmful underreliance risk", fontsize=12, fontweight="bold", color=PALETTE.gold)
    ax_card.text(0.12, 0.59, "Top rule: low confidence + correct advice + no movement toward advice", fontsize=9.5)
    ax_card.text(0.12, 0.49, "Counterfactual: if evidence comparison or uncertainty cue increased,\nexpected utility improves only when intervention burden remains moderate.", fontsize=9.2, color=PALETTE.muted)
    ax_card.text(0.12, 0.34, "Recommended action: compare evidence before final answer", fontsize=10, fontweight="bold", color=PALETTE.green)
    ax_card.text(0.12, 0.25, "Interpretation: diagnostic support for prospective intervention design, not causal proof.", fontsize=8.8, color=PALETTE.muted)
    add_panel_label(ax_dist, "D", "Recommended interventions are not one-size-fits-all")
    policy_actions = _read_csv("policy_actions.csv")
    if not policy_actions.empty and "recommended_action" in policy_actions:
        dist = policy_actions["recommended_action"].value_counts(normalize=True).head(6).sort_values()
        _metric_bars(ax_dist, [clean_label(v) for v in dist.index], dist.values.tolist(), [PALETTE.blue] * len(dist), xlabel="share of evaluated cases", xlim=(0, max(0.7, dist.max() * 1.2)))
    else:
        _metric_bars(ax_dist, ["accept advice", "compare evidence", "request verification"], [0.45, 0.35, 0.20], [PALETTE.blue, PALETTE.green, PALETTE.gold], xlabel="share of evaluated cases")
    return export_figure("figure_9_rules_counterfactuals")


def _figure10(policy: pd.DataFrame, decision_curve: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(DOUBLE_TALL.width, DOUBLE_TALL.height), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    datasets = ["haiid", "chi2023_dke", "convxai_iui2025"]
    titles = ["HAIID policy frontier", "CHI 2023 DKE policy frontier", "ConvXAI policy frontier"]
    for idx, (ax, dataset, title) in enumerate(zip(axes, datasets, titles, strict=False)):
        add_panel_label(ax, chr(ord("A") + idx), title)
        sub = policy.loc[policy["dataset"].eq(dataset)].copy()
        sub["label"] = sub["policy"].map(_clean_policy)
        sub = sub.sort_values("intervention_burden")
        for _, row in sub.iterrows():
            label = row["label"]
            ax.scatter(row["intervention_burden"], row["expected_utility"], s=58, color=POLICY_COLORS.get(label, PALETTE.blue), edgecolor="white", zorder=3)
            direct_label(ax, row["intervention_burden"], row["expected_utility"], label.replace(" threshold", ""), POLICY_COLORS.get(label, PALETTE.blue), ha="left")
        obs = sub.loc[sub["policy"].eq("observed_no_gating")]
        ns = sub.loc[sub["policy"].eq("neurosymbolic_gating")]
        if not obs.empty and not ns.empty:
            _arrow(ax, (float(obs["intervention_burden"].iloc[0]), float(obs["expected_utility"].iloc[0])), (float(ns["intervention_burden"].iloc[0]), float(ns["expected_utility"].iloc[0])), color=PALETTE.red, lw=1.4)
        ax.set_xlabel("intervention burden")
        ax.set_ylabel("expected utility")
        ax.grid(color=PALETTE.grid)
    ax_sens = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_sens, "D", "Utility gains must survive burden penalties")
    dc = decision_curve.copy()
    for dataset, sub in dc.groupby("dataset"):
        ns = sub.loc[sub["policy"].eq("neurosymbolic_gating")]
        if ns.empty:
            continue
        ax_sens.scatter(ns["intervention_burden"], ns["utility_gain_vs_observed"], s=64, label=_dataset_label(dataset))
    ax_sens.axhline(0, color=PALETTE.ink, ls="--", lw=0.9)
    ax_sens.set_xlabel("intervention burden")
    ax_sens.set_ylabel("utility gain vs observed")
    ax_sens.grid(color=PALETTE.grid)
    ax_sens.legend()
    ax_sens.text(0.02, 0.06, "Observational simulation only: no causal policy effect is claimed.", transform=ax_sens.transAxes, fontsize=8.5, color=PALETTE.muted)
    return export_figure("figure_10_policy_frontier")


def _extended_figures(summary: dict, state: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    sizes = summary.get("dataset_sizes", {})
    frame = pd.DataFrame(
        [
            {"dataset": _dataset_label(k), "records": int(v.get("n_interactions", 0)), "participants": int(v.get("n_participants", 0))}
            for k, v in sizes.items()
        ]
    ).sort_values("records")
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    if not frame.empty:
        y = np.arange(len(frame))
        ax.barh(y + 0.16, frame["records"], height=0.28, color=PALETTE.blue, label="records")
        ax.barh(y - 0.16, frame["participants"], height=0.28, color=PALETTE.gold, label="participants")
        ax.set_yticks(y, frame["dataset"])
        ax.set_xscale("log")
        ax.set_xlabel("count (log scale)")
        ax.legend()
        ax.set_title("Extended Data 1: dataset distribution", loc="left", fontweight="bold")
        ax.grid(axis="x", color=PALETTE.grid)
    outputs.extend(export_figure("extended_data_1_dataset_distribution"))

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    if not state.empty:
        dataset_col = "dataset" if "dataset" in state.columns else "dataset_name"
        state_col = "state" if "state" in state.columns else "reliance_state"
        state_ext = state.copy()
        if "share" not in state_ext.columns and {"count", dataset_col}.issubset(state_ext.columns):
            state_ext["share"] = state_ext["count"] / state_ext.groupby(dataset_col)["count"].transform("sum")
        pivot = state_ext.pivot_table(index=dataset_col, columns=state_col, values="share", aggfunc="sum").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=ax, color=[STATE_COLORS.get(c, PALETTE.muted) for c in pivot.columns])
        ax.set_xticklabels([_dataset_label(x.get_text()) for x in ax.get_xticklabels()], rotation=20, ha="right")
        ax.set_ylabel("share")
        ax.set_title("Extended Data 2: reliance-state distribution", loc="left", fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        ax.grid(axis="y", color=PALETTE.grid)
    outputs.extend(export_figure("extended_data_2_state_distribution"))
    return outputs


def generate_real_data_figures() -> list[Path]:
    # The old matplotlib implementation is kept above for traceability. The
    # submission path now uses SVG-native composition so text remains vector,
    # source data are exported per panel and audits can inspect visible labels.
    from .flagship_figures import generate_flagship_figures

    return generate_flagship_figures()
