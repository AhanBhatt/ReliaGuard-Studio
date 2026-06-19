from __future__ import annotations

from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from ..paths import EXPERIMENTS_DIR, PAPER_DIR, PAPER_SOURCE_DATA_DIR, REAL_DATA_EXPERIMENTS_DIR


TABLES_DIR = PAPER_DIR / "tables"


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_table_source(name: str, frame: pd.DataFrame) -> None:
    PAPER_SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PAPER_SOURCE_DATA_DIR / f"{name}.csv", index=False)


def _real_outputs_available() -> bool:
    return (REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv").exists()


def main_model_comparison_table() -> Path:
    if _real_outputs_available():
        frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv")
        split_priority = {"participant": 0, "strict_task": 1, "random": 2}
        frame["split_rank"] = frame["split"].map(split_priority).fillna(9)
        frame = (
            frame.sort_values(["dataset", "target", "split_rank", "auroc", "ece"], ascending=[True, True, True, False, True])
            .groupby(["dataset", "target"])
            .head(1)
            .reset_index(drop=True)
        )
        lines = [
            r"\begin{tabular}{lllcccc}",
            r"\toprule",
            r"Dataset & Target & Split & Model & AUROC & Bal. Acc. & ECE \\",
            r"\midrule",
        ]
        for _, row in frame.iterrows():
            lines.append(
                f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & "
                f"{_escape_latex(_clean_label(row['split']))} & {_escape_latex(_clean_label(row['model']))} & "
                f"{_fmt(row['auroc'])} & {_fmt(row['balanced_accuracy'])} & {_fmt(row['ece'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return _write(TABLES_DIR / "main_model_comparison.tex", "\n".join(lines))

    frame = pd.read_csv(EXPERIMENTS_DIR / "classification_macro_results.csv").head(8)
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & AUROC & AUPRC & F1 & Bal. Acc. & ECE & Brier \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(row['model'])} & "
            f"{_fmt(row['auroc'])} & {_fmt(row['auprc'])} & {_fmt(row['f1'])} & "
            f"{_fmt(row['balanced_accuracy'])} & {_fmt(row['ece'])} & {_fmt(row['brier_score'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "main_model_comparison.tex", "\n".join(lines))


def regression_results_table() -> Path:
    if _real_outputs_available():
        frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "chi2023_miscalibration_effects.csv")
        lines = [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Group & Approp. Reliance & Final Correct & Overreliance & Underreliance \\",
            r"\midrule",
        ]
        for _, row in frame.iterrows():
            lines.append(
                f"{_escape_latex(row['miscalibration_group'])} & {_fmt(row['appropriate_reliance'])} & "
                f"{_fmt(row['final_correct'])} & {_fmt(row['overreliance'])} & {_fmt(row['underreliance'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return _write(TABLES_DIR / "regression_results.tex", "\n".join(lines))

    frame = pd.read_csv(EXPERIMENTS_DIR / "regression_results.csv").sort_values("mae").head(7)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & MAE & RMSE & $R^2$ \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(row['model'])} & {_fmt(row['mae'])} & {_fmt(row['rmse'])} & {_fmt(row['r2'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "regression_results.tex", "\n".join(lines))


def condition_effects_table() -> Path:
    if _real_outputs_available():
        frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "chi2023_condition_effects.csv")
        pivot = frame.pivot(index="condition_name", columns="metric", values="rate").reset_index()
        lines = [
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Condition & Appropriate Reliance & Final Correct & Overreliance \\",
            r"\midrule",
        ]
        for _, row in pivot.iterrows():
            lines.append(
                f"{_escape_latex(row['condition_name'])} & {_fmt(row.get('appropriate_reliance', float('nan')))} & "
                f"{_fmt(row.get('final_correct', float('nan')))} & {_fmt(row.get('overreliance', float('nan')))} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return _write(TABLES_DIR / "condition_effects.tex", "\n".join(lines))

    frame = pd.read_csv(EXPERIMENTS_DIR / "condition_effects.csv").sort_values("overreliance_risk")
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Condition & Immediate & Recall & Transfer & Verif. Fail & Overreliance \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(row['condition_id'])} & "
            f"{_fmt(row['immediate_success'])} & {_fmt(row['delayed_recall_success'])} & "
            f"{_fmt(row['transfer_success'])} & {_fmt(row['verification_failure'])} & "
            f"{_fmt(row['overreliance_risk'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "condition_effects.tex", "\n".join(lines))


def summary_metrics_table() -> Path:
    if _real_outputs_available():
        summary = pd.read_json(REAL_DATA_EXPERIMENTS_DIR / "real_experiment_summary.json", typ="series")
        lines = [
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Statistic & Value \\",
            r"\midrule",
        ]
        for dataset_name, sizes in summary["dataset_sizes"].items():
            lines.append(f"{_escape_latex(_clean_dataset_label(dataset_name))} interactions & {int(sizes['n_interactions'])} \\\\")
            lines.append(f"{_escape_latex(_clean_dataset_label(dataset_name))} participants & {int(sizes['n_participants'])} \\\\")
        lines.extend(
            [
                r"\midrule",
                f"HAIID initial accuracy & {_fmt(summary['haiid_summary']['initial_accuracy'])} \\\\",
                f"HAIID final accuracy & {_fmt(summary['haiid_summary']['final_accuracy'])} \\\\",
                f"HAIID overreliance rate & {_fmt(summary['haiid_summary']['overreliance_rate'])} \\\\",
                f"HAIID underreliance rate & {_fmt(summary['haiid_summary']['underreliance_rate'])} \\\\",
                f"CHI second-batch appropriate reliance & {_fmt(summary['chi2023_summary']['second_batch_appropriate_reliance'])} \\\\",
                f"ConvXAI appropriate reliance & {_fmt(summary['convxai_summary']['appropriate_reliance'])} \\\\",
                f"Pardos mean learning gain & {_fmt(summary.get('pardos_summary', {}).get('mean_learning_gain', float('nan')))} \\\\",
                f"FLoRA mean proposal score & {_fmt(summary.get('flora_summary', {}).get('mean_proposal_score', float('nan')))} \\\\",
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        return _write(TABLES_DIR / "summary_metrics.tex", "\n".join(lines))

    summary = pd.read_json(EXPERIMENTS_DIR / "experiment_summary.json", typ="series")
    best = summary["best_classification_model"]
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
        f"Synthetic participants & {int(summary['n_users'])} \\\\",
        f"Sessions per participant & {int(summary['sessions_per_user'])} \\\\",
        f"Total sessions & {int(summary['total_sessions'])} \\\\",
        f"Best macro model & {_escape_latex(best['model'])} \\\\",
        f"Best macro AUROC & {_fmt(best['auroc'])} \\\\",
        f"Explanation faithfulness & {_fmt(summary['explanation_faithfulness'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return _write(TABLES_DIR / "summary_metrics.tex", "\n".join(lines))


def effect_sizes_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "effect_sizes.csv")
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Contrast & Participants & Effect & 95\% CI \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['contrast']))} & {int(row['n_participants'])} & "
            f"{_fmt(row['effect'])} & [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "effect_sizes.tex", "\n".join(lines))


def mixed_effects_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "clustered_condition_contrasts.csv").head(10)
    lines = [
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"Dataset & Condition & Reference & N & Approp. Reliance & Diff. \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['condition']))} & {_escape_latex(_clean_label(row['reference_condition']))} & "
            f"{int(row['n'])} & {_fmt(row['appropriate_reliance'])} & {_fmt(row['difference_from_reference'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "mixed_effects.tex", "\n".join(lines))


def gee_results_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "gee_results.csv")
    for col in ["estimate", "ci_low", "ci_high", "p_value"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.loc[~frame["term"].isin(["Intercept", "model_not_fit"])].copy()
    frame = frame.loc[frame["estimate"].notna() & (frame["estimate"].abs() < 10)].copy()
    frame = frame.sort_values(["dataset", "outcome", "p_value"]).head(16)
    lines = [
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Dataset & Outcome & Term & Family & Estimate & 95\% CI \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['outcome']))} & {_escape_latex(_clean_label(str(row['term'])[:44]))} & "
            f"{_escape_latex(_clean_label(row['family']))} & {_fmt(row['estimate'])} & [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "gee_results.tex", "\n".join(lines))


def mixed_effects_status_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "mixed_effects_status.csv")
    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Model & Status & Replacement \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(row['model'])} & {_escape_latex(row['status'])} & {_escape_latex(row['replacement'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return _write(TABLES_DIR / "mixed_effects_results.tex", "\n".join(lines))


def split_robustness_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "split_robustness.csv")
    lines = [
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Dataset & Target & Split & Model & AUROC & ECE \\",
        r"\midrule",
    ]
    for _, row in frame.head(18).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & {_escape_latex(_clean_label(row['split']))} & "
            f"{_escape_latex(_clean_label(row['model']))} & {_fmt(row['auroc'])} & {_fmt(row['ece'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "split_robustness.tex", "\n".join(lines))


def cross_dataset_results_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_results.csv")
    frame = frame.loc[frame["auroc"].notna()].copy()
    frame = frame.sort_values(["target", "test_dataset", "auroc"], ascending=[True, True, False]).groupby(["target", "test_dataset"]).head(1)
    lines = [
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Target & Train & Test & Model & AUROC & ECE \\",
        r"\midrule",
    ]
    for _, row in frame.head(14).iterrows():
        lines.append(
            f"{_escape_latex(_clean_label(row['target']))} & {_escape_latex(_clean_dataset_label(row['train_dataset']))} & {_escape_latex(_clean_dataset_label(row['test_dataset']))} & "
            f"{_escape_latex(_clean_label(row['model']))} & {_fmt(row['auroc'])} & {_fmt(row['ece'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "cross_dataset_results.tex", "\n".join(lines))


def calibration_summary_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "calibration_summary.csv")
    frame = frame.sort_values(["dataset", "target", "split", "ece"]).groupby(["dataset", "target", "split"]).head(1)
    lines = [
        r"\begin{tabular}{llllccc}",
        r"\toprule",
        r"Dataset & Target & Split & Model & ECE & MCE & Slope \\",
        r"\midrule",
    ]
    for _, row in frame.head(16).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & {_escape_latex(_clean_label(row['split']))} & "
            f"{_escape_latex(_clean_label(row['model']))} & {_fmt(row['ece'])} & {_fmt(row['mce'])} & {_fmt(row['calibration_slope'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "calibration_summary.tex", "\n".join(lines))


def policy_evaluation_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv")
    frame = frame.sort_values(["dataset", "expected_utility"], ascending=[True, False])
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Policy & Utility & Final correct & Harmful reliance & Burden \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        harmful = row["expected_overreliance"] + row["expected_underreliance"]
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['policy']))} & {_fmt(row['expected_utility'])} & "
            f"{_fmt(row['expected_final_correct'])} & {_fmt(harmful)} & {_fmt(row['intervention_burden'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "policy_evaluation.tex", "\n".join(lines))


def main_policy_evidence_boundary_table() -> Path:
    path = REAL_DATA_EXPERIMENTS_DIR / "policy_evidence_boundary.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    lines = [
        r"\begin{tabular}{@{}p{0.18\linewidth}p{0.22\linewidth}p{0.24\linewidth}p{0.24\linewidth}@{}}",
        r"\toprule",
        r"Dataset & Evidence class & Allowed conclusion & Not allowed \\",
        r"\midrule",
    ]
    for _, row in frame.head(5).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['policy_evidence_class']))} & "
            f"{_escape_latex(row['conclusion_allowed'])} & {_escape_latex(row['conclusion_not_allowed'])} \\\\"
        )
    _write_table_source("table_policy_evidence_boundary", frame)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "main_policy_evidence_boundary.tex", "\n".join(lines))


def reliaguard_summary_table() -> Path:
    path = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_dataset_summary.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Target & Harm in allowed & Missed harmful & Burden & Utility \\",
        r"\midrule",
    ]
    for _, row in frame.head(12).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & "
            f"{_fmt(row['harmful_rate_among_non_intervened'])} & {_fmt(row['missed_harmful_fraction'])} & "
            f"{_fmt(row['intervention_burden'])} & {_fmt(row['expected_utility_proxy'])} \\\\"
        )
    _write_table_source("table_reliaguard_summary", frame)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "reliaguard_summary.tex", "\n".join(lines))


def main_reliaguard_selective_risk_table() -> Path:
    path = REAL_DATA_EXPERIMENTS_DIR / "conformal_selective_risk_results.csv"
    if path.exists():
        frame = pd.read_csv(path)
        focus = frame.loc[
            frame["alpha"].isin([0.05, 0.10, 0.20])
            & frame["comparison_family"].eq("ReliaGuard-NS")
            & frame["split"].isin(["participant", "strict_task", "random"])
        ].copy()
        if focus.empty:
            focus = frame.loc[frame["alpha"].isin([0.05, 0.10, 0.20])].copy()
        focus = (
            focus.sort_values(["dataset", "target", "alpha", "split"])
            .groupby(["dataset", "target", "alpha"], dropna=False)
            .head(1)
            .reset_index(drop=True)
        )
    else:
        focus = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "reliaguard_dataset_summary.csv")
        focus["alpha"] = 0.10
        focus["n_test"] = pd.NA
        focus["threshold"] = pd.NA
        focus["n_calibration"] = pd.NA
        focus["final_correct_among_non_intervened"] = pd.NA
    keep_cols = [
        col
        for col in [
            "dataset",
            "target",
            "alpha",
            "n_calibration",
            "n_test",
            "threshold",
            "empirical_harmful_capture",
            "missed_harmful_fraction",
            "harmful_rate_among_non_intervened",
            "non_intervention_rate",
            "intervention_burden",
            "final_correct_among_non_intervened",
            "expected_utility_proxy",
        ]
        if col in focus.columns
    ]
    _write_table_source("table_reliaguard_selective_risk", focus[keep_cols])
    lines = [
        r"\begin{tabular}{@{}p{0.15\linewidth}p{0.17\linewidth}ccccc@{}}",
        r"\toprule",
        r"Dataset & Target & $\alpha$ & Capture & Missed & Burden & Allowed harm \\",
        r"\midrule",
    ]
    for _, row in focus.head(12).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & "
            f"{_fmt(row['alpha'], 2)} & {_fmt(row['empirical_harmful_capture'])} & {_fmt(row['missed_harmful_fraction'])} & "
            f"{_fmt(row['intervention_burden'])} & {_fmt(row['harmful_rate_among_non_intervened'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "main_reliaguard_selective_risk.tex", "\n".join(lines))


def supplementary_sensitivity_table() -> Path:
    paths = [
        REAL_DATA_EXPERIMENTS_DIR / "sensitivity" / "alpha_sensitivity.csv",
        REAL_DATA_EXPERIMENTS_DIR / "sensitivity" / "policy_burden_sensitivity.csv",
        REAL_DATA_EXPERIMENTS_DIR / "sensitivity" / "label_definition_sensitivity.csv",
        REAL_DATA_EXPERIMENTS_DIR / "sensitivity" / "calibration_sensitivity.csv",
        REAL_DATA_EXPERIMENTS_DIR / "sensitivity" / "harm_weight_sensitivity.csv",
        REAL_DATA_EXPERIMENTS_DIR / "sensitivity" / "threshold_sensitivity.csv",
    ]
    rows = []
    for path in paths:
        display_artifact = _clean_label(path.stem) + " CSV"
        if path.exists():
            frame = pd.read_csv(path)
            rows.append(
                {
                    "analysis": path.stem.replace("_", " "),
                    "rows": len(frame),
                    "artifact": display_artifact,
                    "status": "generated",
                }
            )
        else:
            rows.append({"analysis": path.stem.replace("_", " "), "rows": 0, "artifact": display_artifact, "status": "missing"})
    frame = pd.DataFrame(rows)
    _write_table_source("table_supplementary_sensitivity_inventory", frame)
    lines = [
        r"\begin{tabular}{lrlp{0.30\linewidth}}",
        r"\toprule",
        r"Analysis & Rows & Status & Artifact \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_label(row['analysis']))} & {int(row['rows'])} & {_escape_latex(row['status'])} & {_escape_latex(row['artifact'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "supplementary_sensitivity.tex", "\n".join(lines))


def pardos_learning_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "pardos_learning_effects.csv")
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & N & Pre-test & Post-test & Learning gain \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_label(row['condition_name']))} & {int(row['n'])} & {_fmt(row['pre_test_score'])} & "
            f"{_fmt(row['post_test_score'])} & {_fmt(row['learning_gain'])} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "pardos_learning.tex", "\n".join(lines))


def flora_process_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "flora_process_effects.csv")
    frame = frame.sort_values("score_difference_high_minus_low", ascending=False).head(8)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Process feature & Mean value & Score diff. & 95\% CI \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_label(row['process_feature']))} & {_fmt(row['mean_feature_value'])} & "
            f"{_fmt(row['score_difference_high_minus_low'])} & [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "flora_process.tex", "\n".join(lines))


def flora_process_models_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "flora_process_models.csv")
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & MAE & RMSE & $R^2$ & 95\% CI for $R^2$ \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_label(row['model']))} & {_fmt(row['mae'])} & {_fmt(row['rmse'])} & "
            f"{_fmt(row['r2'])} & [{_fmt(row['r2_ci_low'])}, {_fmt(row['r2_ci_high'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "flora_process_models.tex", "\n".join(lines))


def ablation_summary_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "ablation_summary.csv")
    lines = [
        r"\begin{tabular}{lllcc}",
        r"\toprule",
        r"Dataset & Target & Rule group & Mean AUROC $\Delta$ & Splits \\",
        r"\midrule",
    ]
    for _, row in frame.head(14).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & {_escape_latex(_clean_label(row['group']))} & "
            f"{_fmt(row['mean_auroc_delta'])} & {int(row['n_splits'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "ablation_summary.tex", "\n".join(lines))


def dataset_construct_matrix_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "construct_matrix.csv")
    keep = ["dataset_key", "decision", "appropriate reliance", "overreliance", "underreliance", "confidence shift", "process traces", "learning gain", "delayed recall", "transfer"]
    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Dataset & Decision & Approp. & Over & Under & Conf. & Process & Learn & Recall & Transfer \\",
        r"\midrule",
    ]
    for _, row in frame[keep].iterrows():
        values = [str(int(row[col])) for col in keep[2:]]
        lines.append(f"{_escape_latex(_clean_dataset_label(row['dataset_key']))} & {_escape_latex(_clean_label(row['decision']))} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return _write(TABLES_DIR / "dataset_construct_matrix.tex", "\n".join(lines))


def _clean_dataset_label(text: str) -> str:
    return {
        "haiid": "HAIID",
        "chi2023_dke": "CHI 2023 DKE",
        "convxai_iui2025": "ConvXAI",
        "pardos_chatgpt_tutoring": "Pardos/Bhandari",
        "flora_ips": "FLoRA IPS",
    }.get(str(text), str(text).replace("_", " "))


def _clean_label(text: str) -> str:
    raw = str(text)
    exact = {
        "appropriate_reliance": "Appropriate reliance",
        "overreliance": "Overreliance",
        "underreliance": "Underreliance",
        "harmful_reliance": "Harmful reliance",
        "final_correct": "Final correctness",
        "initial_correct": "Initial correctness",
        "initial_confidence": "Initial confidence",
        "final_confidence": "Final confidence",
        "confidence_shift": "Confidence shift",
        "learning_gain": "Learning gain",
        "proposal_score": "Proposal score",
        "strict_task": "Strict task",
        "participant": "Participant split",
        "random": "Random split",
        "leave_domain_out": "Leave-domain-out",
        "reliance_state_neurosymbolic": "Reliance-state neuro-symbolic",
        "uncertainty_aware_fusion": "Uncertainty-aware fusion",
        "weighted_fusion": "Weighted fusion",
        "learned_fusion": "Learned fusion",
        "calibrated_gradient_boosting": "Calibrated gradient boosting",
        "hist_gradient_boosting": "Histogram gradient boosting",
        "histogram_gradient_boosting": "Histogram gradient boosting",
        "gradient_boosting": "Gradient boosting",
        "calibrated_logistic_regression": "Calibrated logistic regression",
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "symbolic_only": "Symbolic-only",
        "tabular_only": "Tabular-only",
        "no_rules": "No symbolic rules",
        "no_uncertainty": "No uncertainty",
        "no_confidence": "No confidence features",
        "no_domain": "No domain features",
        "observed_no_gating": "Observed/no gating",
        "always_show_advice": "Always show advice",
        "confidence_threshold_gating": "Confidence-threshold gating",
        "symbolic_rule_gating": "Symbolic-rule gating",
        "neurosymbolic_gating": "Neuro-symbolic gating",
        "reliaguard_ns": "ReliaGuard-NS",
        "inappropriate_reliance": "Inappropriate reliance",
        "beneficial_ai_reliance": "Beneficial AI reliance",
        "correct_self_reliance": "Correct self-reliance",
        "independent_correct": "Independent correct",
        "independent_incorrect": "Independent incorrect",
        "harmful_overreliance": "Harmful overreliance",
        "harmful_underreliance": "Harmful underreliance",
        "uncertain_disagreement": "Uncertain disagreement",
    }
    if raw in exact:
        return exact[raw]
    term = raw.replace("\\_", "_")
    categorical = re.match(r"C\((?P<feature>[^)]+)\)\[T\.(?P<level>.+)\]", term)
    if categorical:
        feature = _clean_label(categorical.group("feature"))
        level = _clean_label(categorical.group("level"))
        return f"{feature}: {level}"
    term = term.replace("_", " ").strip()
    term = re.sub(r"\bxai\b", "XAI", term, flags=re.IGNORECASE)
    term = re.sub(r"\bai\b", "AI", term, flags=re.IGNORECASE)
    term = re.sub(r"\bllm\b", "LLM", term, flags=re.IGNORECASE)
    term = re.sub(r"\bgenai\b", "GenAI", term, flags=re.IGNORECASE)
    term = re.sub(r"\bchatgpt\b", "ChatGPT", term, flags=re.IGNORECASE)
    term = term.replace("Llm-Agent", "LLM-agent")
    term = term.replace("Xai", "XAI").replace("Ai", "AI").replace("Undrreliance", "Underreliance")
    term = term.replace("chatxaiAuto", "LLM-agent conversational XAI")
    term = term.replace("chatxaiBoost", "Boosted conversational XAI")
    if not term:
        return raw
    return term[:1].upper() + term[1:]


def main_dataset_summary_table() -> Path:
    summary = pd.read_json(REAL_DATA_EXPERIMENTS_DIR / "real_experiment_summary.json", typ="series")
    construct = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "construct_matrix.csv").set_index("dataset_key")
    supported_cols = [
        "appropriate reliance",
        "overreliance",
        "underreliance",
        "confidence shift",
        "learning gain",
        "process traces",
    ]
    lines = [
        r"\begin{tabular}{p{0.24\linewidth}rrp{0.43\linewidth}}",
        r"\toprule",
        r"Dataset & Records & People & Real-data construct support \\",
        r"\midrule",
    ]
    for dataset, sizes in summary["dataset_sizes"].items():
        supported = [col for col in supported_cols if dataset in construct.index and int(construct.loc[dataset, col]) == 1]
        lines.append(
            f"{_escape_latex(_clean_dataset_label(dataset))} & {int(sizes['n_interactions']):,} & {int(sizes['n_participants']):,} & "
            f"{_escape_latex(', '.join(_clean_label(col) for col in supported) or 'outcome records only')} \\\\"
        )
    _write_table_source(
        "table_1_dataset_evidence_summary",
        pd.DataFrame(
            [
                {
                    "dataset": _clean_dataset_label(dataset),
                    "records": int(sizes["n_interactions"]),
                    "people": int(sizes["n_participants"]),
                    "construct_support": ", ".join(_clean_label(col) for col in supported_cols if dataset in construct.index and int(construct.loc[dataset, col]) == 1),
                }
                for dataset, sizes in summary["dataset_sizes"].items()
            ]
        ),
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "main_dataset_summary.tex", "\n".join(lines))


def main_effect_sizes_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "effect_sizes.csv")
    frame["contrast_type"] = frame["dataset"].map(lambda value: "Observational process contrast" if value == "flora_ips" else "Pre/post or initial/final contrast")
    _write_table_source("table_2_headline_outcome_reliance_contrasts", frame)
    lines = [
        r"\begin{tabular}{@{}p{0.19\linewidth}p{0.23\linewidth}p{0.17\linewidth}rp{0.19\linewidth}@{}}",
        r"\toprule",
        r"Dataset & Contrast & Contrast type & N & Effect (95\% CI) \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['contrast']))} & "
            f"{_escape_latex(row['contrast_type'])} & {int(row['n_participants']):,} & {_fmt(row['effect'])} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "main_effect_sizes.tex", "\n".join(lines))


def main_gee_headline_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "gee_results.csv")
    selections = [
        ("haiid", "overreliance", "C(advice_source_label)[T.human]", "Human- vs AI-labelled advice"),
        ("haiid", "underreliance", "C(advice_source_label)[T.human]", "Human- vs AI-labelled advice"),
        ("haiid", "overreliance", "initial_confidence", "Initial confidence"),
        ("haiid", "underreliance", "initial_confidence", "Initial confidence"),
        ("chi2023_dke", "appropriate_reliance", "first_batch_overestimation", "Self-overestimation"),
        ("chi2023_dke", "appropriate_reliance", "trust_first", "Initial trust"),
        ("convxai_iui2025", "appropriate_reliance", "post_explain_reliability", "Post-explanation reliability"),
        ("pardos_chatgpt_tutoring", "learning_gain", "C(condition_name)[T.No-hint control]", "No-hint vs ChatGPT"),
        ("pardos_chatgpt_tutoring", "learning_gain", "C(condition_name)[T.Human tutor]", "Human tutor vs ChatGPT"),
    ]
    rows = []
    for dataset, outcome, term, label in selections:
        subset = frame.loc[(frame["dataset"] == dataset) & (frame["outcome"] == outcome) & (frame["term"] == term)]
        if not subset.empty:
            row = subset.iloc[0].copy()
            row["label"] = label
            rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table["odds_ratio"] = table.apply(lambda row: float("nan") if str(row["family"]).lower() != "binomial" else float(np.exp(row["estimate"])), axis=1)
        table["or_low"] = table.apply(lambda row: float("nan") if str(row["family"]).lower() != "binomial" else float(np.exp(row["ci_low"])), axis=1)
        table["or_high"] = table.apply(lambda row: float("nan") if str(row["family"]).lower() != "binomial" else float(np.exp(row["ci_high"])), axis=1)
    _write_table_source("table_3_headline_gee_associations", table)
    lines = [
        r"\begin{tabular}{@{}p{0.18\linewidth}p{0.14\linewidth}p{0.20\linewidth}p{0.18\linewidth}p{0.18\linewidth}@{}}",
        r"\toprule",
        r"Dataset & Outcome & Association & Estimate (95\% CI) & OR (95\% CI) \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        or_text = "--"
        if str(row.get("family", "")).lower() == "binomial":
            or_text = f"{_fmt(row['odds_ratio'])} [{_fmt(row['or_low'])}, {_fmt(row['or_high'])}]"
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['outcome']))} & "
            f"{_escape_latex(row['label'])} & {_fmt(row['estimate'])} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] & {_escape_latex(or_text)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "main_gee_headline.tex", "\n".join(lines))


def main_model_policy_summary_table() -> Path:
    split = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "split_robustness.csv")
    cross = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_results.csv")
    model_rows = [
        split.loc[
            (split["dataset"] == "convxai_iui2025")
            & (split["target"] == "overreliance")
            & (split["split"] == "participant")
        ].sort_values("auroc", ascending=False).head(1),
        split.loc[
            (split["dataset"] == "chi2023_dke")
            & (split["target"] == "appropriate_reliance")
            & (split["split"] == "participant")
        ].sort_values("auroc", ascending=False).head(1),
        cross.loc[
            (cross["target"] == "appropriate_reliance")
            & (cross["train_dataset"] == "haiid")
            & (cross["test_dataset"] == "chi2023_dke")
        ].sort_values("auroc", ascending=False).head(1),
    ]
    source_rows = []
    lines = [
        r"\begin{tabular}{@{}p{0.18\linewidth}p{0.32\linewidth}p{0.24\linewidth}p{0.18\linewidth}@{}}",
        r"\toprule",
        r"Validation setting & Dataset / target & Headline result & Interpretation \\",
        r"\midrule",
    ]
    for frame in model_rows:
        if frame.empty:
            continue
        row = frame.iloc[0]
        if "dataset" in row.index:
            analysis_type = "Strict validation"
            dataset_target = f"{_clean_dataset_label(row['dataset'])}: {_clean_label(row['target'])}"
        else:
            analysis_type = "External transfer"
            dataset_target = f"{_clean_dataset_label(row['train_dataset'])} $\\rightarrow$ {_clean_dataset_label(row['test_dataset'])}: {_clean_label(row['target'])}"
        result = f"AUROC {_fmt(row['auroc'])} [{_fmt(row['auroc_ci_low'])}, {_fmt(row['auroc_ci_high'])}]"
        interpretation = "Strict split or external transfer; weak transfers retained."
        source_rows.append(
            {
                "validation_setting": analysis_type,
                "dataset_target": dataset_target.replace("$\\rightarrow$", "to"),
                "headline_result": result,
                "interpretation": interpretation,
            }
        )
        lines.append(f"{_escape_latex(analysis_type)} & {dataset_target} & {_escape_latex(result)} & {_escape_latex(interpretation)} \\\\")
    _write_table_source("table_4_method_policy_summary", pd.DataFrame(source_rows))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write(TABLES_DIR / "main_model_policy_summary.tex", "\n".join(lines))


def extended_model_results_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv")
    frame = frame.sort_values(["dataset", "target", "split", "auroc"], ascending=[True, True, True, False]).groupby(["dataset", "target", "split"]).head(2)
    lines = [
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllccc}",
        r"\toprule",
        r"Dataset & Target & Split & Model & AUROC & AUPRC & ECE \\",
        r"\midrule",
    ]
    for _, row in frame.head(36).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & {_escape_latex(_clean_label(row['split']))} & "
            f"{_escape_latex(_clean_label(row['model']))} & {_fmt(row['auroc'])} & {_fmt(row['auprc'])} & {_fmt(row['ece'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return _write(TABLES_DIR / "extended_model_results.tex", "\n".join(lines))


def extended_calibration_results_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "calibration_summary.csv")
    frame = frame.sort_values(["dataset", "target", "split", "ece"]).groupby(["dataset", "target", "split"]).head(2)
    lines = [
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllcccc}",
        r"\toprule",
        r"Dataset & Target & Split & Model & ECE & MCE & Brier & Slope \\",
        r"\midrule",
    ]
    for _, row in frame.head(36).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & {_escape_latex(_clean_label(row['split']))} & "
            f"{_escape_latex(_clean_label(row['model']))} & {_fmt(row['ece'])} & {_fmt(row['mce'])} & {_fmt(row['brier_score'])} & {_fmt(row['calibration_slope'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return _write(TABLES_DIR / "extended_calibration_results.tex", "\n".join(lines))


def extended_rule_ablation_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "ablation_summary.csv")
    lines = [
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"Dataset & Target & Rule group & Mean $\Delta$AUROC & 95\% CI & Splits \\",
        r"\midrule",
    ]
    for _, row in frame.head(30).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['target']))} & {_escape_latex(_clean_label(row['group']))} & "
            f"{_fmt(row['mean_auroc_delta'])} & [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] & {int(row['n_splits'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return _write(TABLES_DIR / "extended_rule_ablation.tex", "\n".join(lines))


def extended_cross_dataset_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "cross_dataset_results.csv")
    frame = frame.sort_values(["target", "test_dataset", "auroc"], ascending=[True, True, False]).groupby(["target", "test_dataset"]).head(2)
    lines = [
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllccc}",
        r"\toprule",
        r"Target & Train & Test & Model & AUROC & AUPRC & ECE \\",
        r"\midrule",
    ]
    for _, row in frame.head(36).iterrows():
        lines.append(
            f"{_escape_latex(_clean_label(row['target']))} & {_escape_latex(_clean_dataset_label(row['train_dataset']))} & {_escape_latex(_clean_dataset_label(row['test_dataset']))} & "
            f"{_escape_latex(_clean_label(row['model']))} & {_fmt(row['auroc'])} & {_fmt(row['auprc'])} & {_fmt(row['ece'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return _write(TABLES_DIR / "extended_cross_dataset.tex", "\n".join(lines))


def extended_flora_process_table() -> Path:
    frame = pd.read_csv(REAL_DATA_EXPERIMENTS_DIR / "flora_process_effects.csv")
    lines = [
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Feature & Mean & High-score mean & Low-score mean & Difference (95\% CI) \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape_latex(_clean_label(row['process_feature']))} & {_fmt(row['mean_feature_value'])} & {_fmt(row['high_feature_mean_score'])} & "
            f"{_fmt(row['low_feature_mean_score'])} & {_fmt(row['score_difference_high_minus_low'])} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return _write(TABLES_DIR / "extended_flora_process.tex", "\n".join(lines))


def extended_sensitivity_results_table() -> Path:
    path = REAL_DATA_EXPERIMENTS_DIR / "label_definition_sensitivity.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    lines = [
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Dataset & Outcome & Definition & N & Rate & 95\% CI \\",
        r"\midrule",
    ]
    for _, row in frame.head(24).iterrows():
        lines.append(
            f"{_escape_latex(_clean_dataset_label(row['dataset']))} & {_escape_latex(_clean_label(row['outcome']))} & {_escape_latex(_clean_label(row['definition']))} & "
            f"{int(row['n'])} & {_fmt(row['rate'])} & [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return _write(TABLES_DIR / "extended_sensitivity_results.tex", "\n".join(lines))


def generate_paper_tables() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if _real_outputs_available():
        outputs["main_dataset_summary"] = main_dataset_summary_table()
        outputs["main_effect_sizes"] = main_effect_sizes_table()
        outputs["main_gee_headline"] = main_gee_headline_table()
        outputs["main_model_policy_summary"] = main_model_policy_summary_table()
        outputs["chi_miscalibration_results"] = TABLES_DIR / "regression_results.tex"
        outputs["effect_sizes"] = effect_sizes_table()
        outputs["mixed_effects"] = mixed_effects_table()
        outputs["gee_results"] = gee_results_table()
        outputs["mixed_effects_results"] = mixed_effects_status_table()
        outputs["split_robustness"] = split_robustness_table()
        outputs["cross_dataset_results"] = cross_dataset_results_table()
        outputs["calibration_summary"] = calibration_summary_table()
        outputs["policy_evaluation"] = policy_evaluation_table()
        outputs["main_policy_evidence_boundary"] = main_policy_evidence_boundary_table()
        outputs["reliaguard_summary"] = reliaguard_summary_table()
        outputs["main_reliaguard_selective_risk"] = main_reliaguard_selective_risk_table()
        outputs["pardos_learning"] = pardos_learning_table()
        outputs["flora_process"] = flora_process_table()
        outputs["flora_process_models"] = flora_process_models_table()
        outputs["ablation_summary"] = ablation_summary_table()
        outputs["dataset_construct_matrix"] = dataset_construct_matrix_table()
        outputs["extended_model_results"] = extended_model_results_table()
        outputs["extended_calibration_results"] = extended_calibration_results_table()
        outputs["extended_rule_ablation"] = extended_rule_ablation_table()
        outputs["extended_cross_dataset"] = extended_cross_dataset_table()
        outputs["extended_flora_process"] = extended_flora_process_table()
        outputs["extended_sensitivity_results"] = extended_sensitivity_results_table()
        outputs["supplementary_sensitivity"] = supplementary_sensitivity_table()
        aliases = {
            "extended_gee_full": ("gee_results.tex", "extended_gee_full.tex"),
            "extended_model_full": ("extended_model_results.tex", "extended_model_full.tex"),
            "extended_calibration_full": ("extended_calibration_results.tex", "extended_calibration_full.tex"),
            "extended_transfer_full": ("extended_cross_dataset.tex", "extended_transfer_full.tex"),
            "extended_policy_full": ("policy_evaluation.tex", "extended_policy_full.tex"),
            "extended_rule_ablation_full": ("extended_rule_ablation.tex", "extended_rule_ablation_full.tex"),
            "extended_sensitivity_full": ("extended_sensitivity_results.tex", "extended_sensitivity_full.tex"),
            "extended_label_sensitivity": ("extended_sensitivity_results.tex", "extended_label_sensitivity.tex"),
            "extended_policy_sensitivity": ("extended_sensitivity_results.tex", "extended_policy_sensitivity.tex"),
        }
        for key, (source_name, alias_name) in aliases.items():
            source = TABLES_DIR / source_name
            alias = TABLES_DIR / alias_name
            if source.exists():
                try:
                    shutil.copy2(source, alias)
                except PermissionError:
                    alias = TABLES_DIR / alias_name.replace(".tex", "_review.tex")
                    shutil.copy2(source, alias)
                outputs[key] = alias
    else:
        outputs = {
            "legacy_model_comparison": main_model_comparison_table(),
            "legacy_regression_results": regression_results_table(),
            "legacy_condition_effects": condition_effects_table(),
            "legacy_summary_metrics": summary_metrics_table(),
        }
    return outputs
