from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

from .config.loader import load_project_config
from .data.dataset_screening import write_dataset_screening
from .data.adapters.registry import get_dataset_registry, render_registry_markdown
from .evaluation.claims_audit import audit_claims
from .evaluation.cross_dataset import run_cross_dataset_summary
from .evaluation.cross_dataset_generalization import run_cross_dataset_generalization
from .evaluation.leakage_audit import run_leakage_audit
from .evaluation.negative_controls import run_negative_controls
from .evaluation.nmi_submission_audit import run_nmi_submission_audit
from .evaluation.policy_evaluation import run_policy_evaluation
from .evaluation.policy_value_bounds import run_policy_bound_suite
from .evaluation.real_data_runner import download_real_data, prepare_real_data, run_real_experiments
from .evaluation.reporting_standard import write_reporting_standard
from .evaluation.runner import run_full_experiment
from .evaluation.selective_risk_guarantees import run_conformal_risk_control
from .evaluation.coverage_utility_tradeoff import write_coverage_utility_tradeoff
from .evaluation.conformal_ablation import write_conformal_ablation
from .evaluation.conformal_sensitivity import run_sensitivity_analyses
from .paths import REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR, REPO_ROOT
from .visualization.figure_quality_audit import run_figure_quality_audit
from .visualization.pdf_render_audit import run_visual_feedback_loop
from .visualization.real_data_figures import generate_real_data_figures
from .visualization.render_safety import audit_render_safety
from .visualization.figures import generate_all_figures
from .visualization.tables import generate_paper_tables
from .visualization.visual_text_audit import run_visual_text_audit


app = typer.Typer(help="ReliaGuard Studio CLI")


@app.command("run-experiments")
def run_experiments(config_path: Path | None = typer.Option(None, help="Optional path to a config YAML file.")) -> None:
    config = load_project_config(config_path)
    artifacts = run_full_experiment(config)
    typer.echo(f"Datasets written to: {artifacts.dataset_paths}")
    typer.echo(f"Experiment outputs written to: {artifacts.experiment_paths}")


@app.command("generate-figures")
def figures() -> None:
    outputs = generate_all_figures()
    typer.echo(f"Generated figures: {outputs}")


@app.command("search-datasets")
def search_datasets() -> None:
    registry = get_dataset_registry()
    typer.echo(render_registry_markdown(registry.values()))


@app.command("screen-datasets")
def screen_datasets() -> None:
    outputs = write_dataset_screening()
    typer.echo(f"Dataset-screening outputs refreshed: {outputs}")


@app.command("write-reporting-standard")
def cli_write_reporting_standard() -> None:
    outputs = write_reporting_standard()
    typer.echo(f"Reliance-state reporting-standard outputs refreshed: {outputs}")


@app.command("download-real-data")
def cli_download_real_data(force: bool = typer.Option(False, help="Re-download auto-download datasets.")) -> None:
    outputs = download_real_data(force=force)
    typer.echo(f"Real-data download outputs: {outputs}")


@app.command("prepare-real-data")
def cli_prepare_real_data(force_download: bool = typer.Option(False, help="Force download before preparation.")) -> None:
    datasets = prepare_real_data(force_download=force_download)
    typer.echo(f"Prepared datasets: {list(datasets.keys())}")


@app.command("run-real-experiments")
def cli_run_real_experiments(force: bool = typer.Option(False, help="Force a full recomputation instead of reusing complete cached artifacts.")) -> None:
    artifacts = run_real_experiments(force=force)
    typer.echo(f"Prepared datasets: {artifacts.prepared_paths}")
    typer.echo(f"Real experiment outputs: {artifacts.experiment_paths}")


@app.command("run-cross-dataset")
def cli_run_cross_dataset() -> None:
    model_results_path = REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv"
    if not model_results_path.exists():
        run_real_experiments()
    prepared_interactions: dict[str, pd.DataFrame] = {}
    for dataset_dir in REAL_DATA_PREPARED_DIR.iterdir():
        interactions_path = dataset_dir / "interactions.csv"
        if interactions_path.exists():
            prepared_interactions[dataset_dir.name] = pd.read_csv(interactions_path, low_memory=False)
    if not prepared_interactions:
        run_real_experiments()
        for dataset_dir in REAL_DATA_PREPARED_DIR.iterdir():
            interactions_path = dataset_dir / "interactions.csv"
            if interactions_path.exists():
                prepared_interactions[dataset_dir.name] = pd.read_csv(interactions_path, low_memory=False)
    outputs = run_cross_dataset_summary(prepared_interactions, pd.read_csv(model_results_path))
    outputs.update(run_cross_dataset_generalization(prepared_interactions))
    typer.echo(f"Cross-dataset outputs refreshed: {outputs}")


@app.command("run-policy-evaluation")
def cli_run_policy_evaluation() -> None:
    outputs = run_policy_evaluation()
    typer.echo(f"Policy evaluation outputs refreshed: {outputs}")


@app.command("run-conformal-risk-control")
def cli_run_conformal_risk_control(alpha: float = typer.Option(0.10, min=0.01, max=0.50, help="Target missed-harmful level for conformal control.")) -> None:
    if not (REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv").exists():
        run_real_experiments()
    outputs = run_conformal_risk_control(alpha=alpha)
    outputs.update(write_coverage_utility_tradeoff())
    outputs.update(write_conformal_ablation())
    typer.echo(f"ReliaGuard-NS conformal outputs refreshed: {outputs}")


@app.command("run-off-policy-evaluation")
def cli_run_off_policy_evaluation() -> None:
    if not (REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv").exists():
        run_policy_evaluation()
    outputs = run_policy_bound_suite()
    typer.echo(f"Policy-bound outputs refreshed: {outputs}")


@app.command("run-sensitivity-analyses")
def cli_run_sensitivity_analyses() -> None:
    if not (REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv").exists():
        run_real_experiments()
    if not (REAL_DATA_EXPERIMENTS_DIR / "policy_evaluation.csv").exists():
        run_policy_evaluation()
    outputs = run_sensitivity_analyses()
    typer.echo(f"Sensitivity outputs refreshed: {outputs}")


@app.command("run-negative-controls")
def cli_run_negative_controls() -> None:
    if not (REAL_DATA_EXPERIMENTS_DIR / "real_predictions.csv").exists():
        run_real_experiments()
    outputs = run_negative_controls()
    typer.echo(f"Negative-control outputs refreshed: {outputs}")


@app.command("run-leakage-audit")
def cli_run_leakage_audit() -> None:
    if not (REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv").exists():
        run_real_experiments()
    outputs = run_leakage_audit()
    typer.echo(f"Leakage-audit outputs refreshed: {outputs}")


@app.command("generate-real-figures")
def cli_generate_real_figures() -> None:
    outputs = generate_real_data_figures()
    quality_outputs = run_figure_quality_audit()
    typer.echo(f"Generated real-data figures: {outputs}")
    typer.echo(f"Figure quality outputs: {quality_outputs}")


@app.command("audit-figures")
def cli_audit_figures() -> None:
    outputs = run_figure_quality_audit()
    typer.echo(f"Figure quality outputs: {outputs}")


@app.command("audit-render-safety")
def cli_audit_render_safety() -> None:
    path = audit_render_safety()
    typer.echo(f"Render-safety audit passed: {path}")


@app.command("audit-visual-text")
def cli_audit_visual_text() -> None:
    outputs = run_visual_text_audit()
    typer.echo(f"Visual text audit passed: {outputs}")


@app.command("visual-feedback-loop")
def cli_visual_feedback_loop() -> None:
    outputs = run_visual_feedback_loop()
    typer.echo(f"Visual feedback loop outputs: {outputs}")


@app.command("audit-claims")
def cli_audit_claims() -> None:
    import json

    summary_path = REPO_ROOT / "artifacts" / "real_data" / "experiments" / "real_experiment_summary.json"
    if not summary_path.exists():
        run_real_experiments()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    path = audit_claims(summary)
    typer.echo(f"Claims checklist written to: {path}")


@app.command("nmi-submission-audit")
def cli_nmi_submission_audit() -> None:
    outputs = run_nmi_submission_audit(run_subprocess_checks=False)
    typer.echo(f"NMI submission audit passed: {outputs}")


@app.command("build-paper")
def build_paper() -> None:
    paper_dir = REPO_ROOT / "paper"
    tex = paper_dir / "main.tex"
    if not tex.exists():
        raise typer.Exit("paper/main.tex does not exist yet.")
    if not (REPO_ROOT / "artifacts" / "real_data" / "experiments" / "real_experiment_summary.json").exists():
        run_real_experiments()
    if not (paper_dir / "figures" / "figure_01_evidence_to_action_map.pdf").exists():
        try:
            generate_real_data_figures()
        except Exception:
            generate_all_figures()
    generate_paper_tables()
    target = paper_dir / "main.pdf"
    jobname = "main"
    try:
        with target.open("ab"):
            pass
    except OSError:
        jobname = "main_review"
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={jobname}", "-output-directory", str(paper_dir), str(tex)],
            check=True,
            cwd=REPO_ROOT,
        )
    subprocess.run(
        ["bibtex", jobname],
        check=False,
        cwd=paper_dir,
    )
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={jobname}", "-output-directory", str(paper_dir), str(tex)],
            check=True,
            cwd=REPO_ROOT,
        )
    typer.echo(f"Built paper PDF at {paper_dir / f'{jobname}.pdf'}")


@app.command("build-supplement")
def build_supplement() -> None:
    paper_dir = REPO_ROOT / "paper"
    supplement_dir = paper_dir / "supplementary"
    tex = supplement_dir / "main_supplement.tex"
    if not tex.exists():
        raise typer.Exit("paper/supplementary/main_supplement.tex does not exist yet.")
    generate_paper_tables()
    supplement_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(supplement_dir), str(tex)],
            check=True,
            cwd=paper_dir,
        )
    built = supplement_dir / "main_supplement.pdf"
    root_copy = paper_dir / "supplementary.pdf"
    if built.exists():
        root_copy.write_bytes(built.read_bytes())
    typer.echo(f"Built supplementary PDF at {built}")


@app.command("serve-api")
def serve_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "reliaguard_studio.api.main:app", "--host", host, "--port", str(port)],
        check=True,
        cwd=REPO_ROOT,
    )


@app.command("serve-study")
def serve_study(host: str = "127.0.0.1", port: int = 8010) -> None:
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "reliaguard_studio.study_platform.app:app", "--host", host, "--port", str(port)],
        check=True,
        cwd=REPO_ROOT,
    )


@app.command("simulate-study")
def simulate_study(n: int = typer.Option(120, min=12, help="Number of synthetic pilot participants for platform smoke testing.")) -> None:
    from .study_platform.simulation import simulate_prospective_trial

    outputs = simulate_prospective_trial(n_participants=n)
    typer.echo(f"Simulated prospective-study smoke-test data: {outputs}")


@app.command("export-study-data")
def export_study_data() -> None:
    from .study_platform.export import export_study_data

    outputs = export_study_data()
    typer.echo(f"Exported prospective-study data: {outputs}")


@app.command("analyze-study-data")
def analyze_study_data() -> None:
    from .study_platform.analysis import analyze_study_data

    outputs = analyze_study_data()
    typer.echo(f"Analyzed prospective-study data: {outputs}")


@app.command("power-study")
def power_study() -> None:
    from .study_platform.power import write_power_analysis

    outputs = write_power_analysis()
    typer.echo(f"Prospective-study power analysis written: {outputs}")


@app.command("validate-study-platform")
def validate_study_platform() -> None:
    from .study_platform.validation import validate_study_platform

    outputs = validate_study_platform()
    typer.echo(f"Prospective-study platform validation written: {outputs}")


@app.command("generate-study-report")
def generate_study_report() -> None:
    from .study_platform.report import generate_study_report

    outputs = generate_study_report()
    typer.echo(f"Prospective-study readiness report written: {outputs}")


@app.command("run-all")
def run_all() -> None:
    download_real_data(force=False)
    run_real_experiments()
    run_policy_evaluation()
    run_conformal_risk_control()
    write_coverage_utility_tradeoff()
    write_conformal_ablation()
    run_policy_bound_suite()
    run_sensitivity_analyses()
    run_negative_controls()
    run_leakage_audit()
    write_dataset_screening()
    write_reporting_standard()
    audit_render_safety()
    generate_real_data_figures()
    run_figure_quality_audit()
    cli_audit_claims()
    if (REPO_ROOT / "paper" / "main.tex").exists():
        build_paper()
        build_supplement()
        run_visual_text_audit()
    else:
        typer.echo("Skipping private manuscript build: paper/main.tex is not present in this public checkout.")


if __name__ == "__main__":
    app()
