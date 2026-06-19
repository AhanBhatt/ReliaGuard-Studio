from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from ..paths import DATASETS_DIR, EXPERIMENTS_DIR, PAPER_FIGURES_DIR, ensure_directories


plt.style.use("default")


def _save(fig: plt.Figure, name: str) -> dict[str, Path]:
    ensure_directories()
    target_dir = PAPER_FIGURES_DIR / "synthetic_stress_tests"
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / f"{name}.pdf"
    png_path = target_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def _box(ax, xy, width, height, text, color):
    rect = plt.Rectangle(xy, width, height, facecolor=color, edgecolor="#1f2937", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)


def figure_1_conceptual_framework() -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _box(ax, (0.3, 3.9), 1.7, 1.0, "Simulated\nassistance", "#dbeafe")
    _box(ax, (0.3, 1.8), 1.7, 1.0, "Simulated\nparticipant", "#e0f2fe")
    _box(ax, (2.4, 2.8), 2.0, 1.2, "Synthetic traces\nprompts, edits,\nconfidence", "#ecfccb")
    _box(ax, (4.9, 3.9), 2.0, 1.0, "Sequence and\ntabular models", "#fee2e2")
    _box(ax, (4.9, 1.7), 2.0, 1.0, "Rule engine\nfuzzy + temporal", "#fde68a")
    _box(ax, (7.4, 2.8), 1.8, 1.2, "Neuro-symbolic\nfusion", "#ddd6fe")
    _box(ax, (9.6, 3.9), 2.0, 1.0, "Synthetic outcome\ntargets", "#fecaca")
    _box(ax, (9.6, 1.6), 2.0, 1.2, "Explanations +\ninterventions", "#bbf7d0")
    arrows = [
        ((2.0, 4.4), (2.4, 3.6)),
        ((2.0, 2.3), (2.4, 3.2)),
        ((4.4, 3.4), (4.9, 4.35)),
        ((4.4, 3.2), (4.9, 2.2)),
        ((6.9, 4.35), (7.4, 3.4)),
        ((6.9, 2.2), (7.4, 3.2)),
        ((9.2, 3.4), (9.6, 4.35)),
        ((9.2, 3.0), (9.6, 2.2)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.7, color="#374151"))
    ax.set_title("Supplementary AIR-Bench simulation framework", fontsize=14)
    return _save(fig, "figure_1_conceptual_framework")


def figure_2_system_architecture() -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")
    _box(ax, (0.5, 5.7), 2.0, 1.1, "Frontend demo\n(static + API)", "#dbeafe")
    _box(ax, (0.5, 3.8), 2.0, 1.1, "FastAPI backend", "#bfdbfe")
    _box(ax, (3.0, 6.0), 2.2, 0.9, "Canonical config\nYAML + Pydantic", "#e0f2fe")
    _box(ax, (3.0, 4.5), 2.2, 0.9, "Data layer\nAIR-Bench + adapters", "#cffafe")
    _box(ax, (3.0, 3.0), 2.2, 0.9, "Persistence\nSQLite local-first", "#dcfce7")
    _box(ax, (5.8, 5.5), 2.2, 1.0, "Models\nlogistic, RF, MLP,\nGRU/LSTM, transformer", "#fee2e2")
    _box(ax, (5.8, 3.6), 2.2, 1.0, "Rules\nfuzzy, temporal,\nprobabilistic", "#fde68a")
    _box(ax, (8.5, 4.5), 2.0, 1.0, "Evaluation\nmetrics, calibration,\nrobustness, ablation", "#e9d5ff")
    _box(ax, (8.5, 2.7), 2.0, 1.0, "Visualization\npaper figures,\nreports, exports", "#ddd6fe")
    _box(ax, (10.9, 5.4), 1.7, 1.0, "Experiments", "#fecaca")
    _box(ax, (10.9, 3.1), 1.7, 1.0, "Paper build", "#fecdd3")
    edges = [((2.5, 6.2), (3.0, 6.45)), ((2.5, 4.35), (3.0, 4.95)), ((2.5, 4.35), (3.0, 3.45)),
             ((5.2, 6.45), (5.8, 6.0)), ((5.2, 4.95), (5.8, 4.1)), ((5.2, 3.45), (5.8, 4.1)),
             ((8.0, 6.0), (8.5, 5.0)), ((8.0, 4.1), (8.5, 5.0)), ((8.0, 4.1), (8.5, 3.2)),
             ((10.5, 5.0), (10.9, 5.9)), ((10.5, 3.2), (10.9, 3.6))]
    for start, end in edges:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6, color="#374151"))
    ax.set_title("Supplementary research system architecture", fontsize=14)
    return _save(fig, "figure_2_system_architecture")


def figure_3_benchmark_design(tasks: pd.DataFrame, sessions: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    task_counts = tasks["family"].value_counts().sort_index()
    axes[0].barh(task_counts.index, task_counts.values, color="#38bdf8")
    axes[0].set_title("Task family coverage")
    axes[0].set_xlabel("Synthetic task count")
    condition_counts = sessions["condition_id"].value_counts().sort_index()
    axes[1].bar(condition_counts.index, condition_counts.values, color="#a78bfa")
    axes[1].set_title("Assistance condition allocation")
    axes[1].tick_params(axis="x", rotation=40)
    fig.suptitle("Supplementary AIR-Bench task and condition design", fontsize=14)
    return _save(fig, "figure_3_benchmark_design")


def figure_4_model_diagram() -> dict[str, Path]:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("Session traces", "Temporal neural encoder"),
            ("Session traces", "Rule engine"),
            ("Temporal neural encoder", "Uncertainty estimator"),
            ("Rule engine", "Counterfactual generator"),
            ("Temporal neural encoder", "Fusion"),
            ("Rule engine", "Fusion"),
            ("Uncertainty estimator", "Fusion"),
            ("Fusion", "Outcome predictions"),
            ("Fusion", "Explanation layer"),
            ("Counterfactual generator", "Explanation layer"),
        ]
    )
    pos = {
        "Session traces": (0.0, 0.5),
        "Temporal neural encoder": (1.7, 0.8),
        "Rule engine": (1.7, 0.2),
        "Uncertainty estimator": (3.3, 0.9),
        "Counterfactual generator": (3.3, 0.1),
        "Fusion": (5.0, 0.5),
        "Outcome predictions": (6.8, 0.8),
        "Explanation layer": (6.8, 0.2),
    }
    fig, ax = plt.subplots(figsize=(11, 4.8))
    nx.draw_networkx_nodes(graph, pos, node_color="#e0f2fe", node_size=2600, edgecolors="#1f2937", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=1.8, arrows=True, arrowstyle="-|>", ax=ax)
    ax.set_title("Supplementary neuro-symbolic simulation model", fontsize=14)
    ax.axis("off")
    return _save(fig, "figure_4_neuro_symbolic_model")


def figure_5_main_results(classification_macro: pd.DataFrame, regression_results: pd.DataFrame) -> dict[str, Path]:
    top_models = classification_macro.head(7).copy()
    regression_summary = regression_results.groupby("model")[["mae"]].mean(numeric_only=True).sort_values("mae")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(top_models["model"], top_models["auroc"], color="#34d399")
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_title("Macro AUROC across classification targets")
    axes[0].set_xlabel("AUROC")
    axes[1].barh(regression_summary.index[:7], regression_summary["mae"].iloc[:7], color="#fb7185")
    axes[1].invert_xaxis()
    axes[1].set_title("Retention-gap MAE")
    axes[1].set_xlabel("Lower is better")
    fig.suptitle("Supplementary synthetic stress-test results", fontsize=14)
    return _save(fig, "figure_5_main_results")


def figure_6_calibration(calibration_curve: pd.DataFrame) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect calibration")
    ax.plot(calibration_curve["mean_confidence"], calibration_curve["accuracy"], marker="o", color="#0f766e", label="learned fusion")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Synthetic reliability curve")
    ax.legend()
    return _save(fig, "figure_6_calibration")


def figure_7_rule_analysis(rule_activations: pd.DataFrame, sessions: pd.DataFrame) -> dict[str, Path]:
    merged = rule_activations.merge(sessions[["session_id", "latent_driver"]], on="session_id", how="left")
    heatmap = (
        merged.groupby(["group", "latent_driver"])["activation"]
        .mean()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap.to_numpy(), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(heatmap.columns)), labels=heatmap.columns, rotation=30)
    ax.set_yticks(range(len(heatmap.index)), labels=heatmap.index)
    ax.set_title("Synthetic rule activation by latent driver")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return _save(fig, "figure_7_rule_analysis")


def extended_data_figures(sessions: pd.DataFrame, feature_ablation: pd.DataFrame, robustness: pd.DataFrame, sample_report: dict) -> dict[str, dict[str, Path]]:
    outputs = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sessions["task_family"].value_counts().plot(kind="bar", ax=axes[0], color="#60a5fa", title="Task family distribution")
    sessions["condition_id"].value_counts().plot(kind="bar", ax=axes[1], color="#c084fc", title="Condition distribution")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].tick_params(axis="x", rotation=35)
    outputs["extended_data_1_distribution"] = _save(fig, "extended_data_1_distribution")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(feature_ablation["group"], feature_ablation["auroc"], color="#f59e0b")
    ax.set_title("Synthetic feature-group ablation")
    outputs["extended_data_2_feature_ablation"] = _save(fig, "extended_data_2_feature_ablation")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(robustness["scenario"], robustness["auroc"], color="#ef4444")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Synthetic robustness scenarios")
    ax.tick_params(axis="x", rotation=30)
    outputs["extended_data_3_robustness"] = _save(fig, "extended_data_3_robustness")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axis("off")
    session = sample_report["session"]
    explanation = sample_report["symbolic_explanation"]
    lines = [
        "Example synthetic session report",
        f"Session: {session['session_id']} | Condition: {session['condition_id']} | Family: {session['task_family']}",
        f"Overreliance score: {session['overreliance_score']:.2f} | Verification failure: {session['verification_failure']}",
        f"Retention gap: {session['retention_gap']:.2f} | Transfer gap: {session['transfer_gap']:.2f}",
        f"Top symbolic summary: {explanation['summary']}",
        "Counterfactuals:",
    ] + [f"- {item}" for item in explanation["counterfactuals"]]
    ax.text(0.01, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10, family="monospace")
    outputs["extended_data_4_sample_report"] = _save(fig, "extended_data_4_sample_report")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(
        0.01,
        0.95,
        "Config/schema overview\n"
        "Sections: metadata, simulation, model, targets, indicators, usage_patterns,\n"
        "task_families, assistance_conditions, metrics, rules.\n"
        "All runtime modules load the same canonical YAML source of truth.",
        ha="left",
        va="top",
        fontsize=11,
    )
    outputs["extended_data_5_config_overview"] = _save(fig, "extended_data_5_config_overview")
    return outputs


def generate_all_figures() -> dict[str, dict[str, Path]]:
    tasks = pd.read_csv(DATASETS_DIR / "tasks.csv")
    sessions = pd.read_csv(DATASETS_DIR / "sessions.csv")
    classification_macro = pd.read_csv(EXPERIMENTS_DIR / "classification_macro_results.csv")
    regression_results = pd.read_csv(EXPERIMENTS_DIR / "regression_results.csv")
    calibration_curve = pd.read_csv(EXPERIMENTS_DIR / "calibration_curve.csv")
    rule_activations = pd.read_csv(EXPERIMENTS_DIR / "rule_activations.csv")
    feature_ablation = pd.read_csv(EXPERIMENTS_DIR / "feature_ablation_results.csv")
    robustness = pd.read_csv(EXPERIMENTS_DIR / "robustness_results.csv")
    with (EXPERIMENTS_DIR / "sample_session_report.json").open("r", encoding="utf-8") as handle:
        sample_report = json.load(handle)

    outputs = {
        "figure_1": figure_1_conceptual_framework(),
        "figure_2": figure_2_system_architecture(),
        "figure_3": figure_3_benchmark_design(tasks, sessions),
        "figure_4": figure_4_model_diagram(),
        "figure_5": figure_5_main_results(classification_macro, regression_results),
        "figure_6": figure_6_calibration(calibration_curve),
        "figure_7": figure_7_rule_analysis(rule_activations, sessions),
    }
    outputs.update(extended_data_figures(sessions, feature_ablation, robustness, sample_report))
    return outputs
