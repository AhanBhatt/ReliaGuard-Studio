# ReliaGuard Studio

**A production-style AI evaluation platform for detecting overreliance, underreliance, and unsafe human-AI decision behavior.**

AI assistants can improve average accuracy while still creating harmful reliance failures: people may accept wrong advice, reject correct advice, or become poorly calibrated. ReliaGuard Studio turns those hidden failure modes into measurable product signals: reliance-state probabilities, calibrated risk, symbolic rule traces, counterfactual explanations, conformal thresholds, and policy-simulation outputs.

The underlying analysis integrates **43,263 public records from 2,229 participants/students across five datasets**: HAIID, CHI 2023 DKE, ConvXAI, Pardos/Bhandari ChatGPT tutoring, and FLoRA IPS.

## Product Demo Flow

1. Open the landing page and see the core thesis in 30 seconds.
2. Try the interactive case simulator.
3. Inspect the rule-trace explanation and counterfactual.
4. Move the conformal alpha slider to see capture-versus-burden trade-offs.
5. Compare baselines, calibration metrics, and cross-dataset transfer.
6. Review policy-simulation outputs for no gating, confidence-threshold gating, symbolic rules, and ReliaGuard-NS.

For a complete walkthrough of setup, app pages, API calls, interpretation boundaries and portfolio talking points, see [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md).

## Architecture

```text
apps/web                 Next.js + TypeScript + Tailwind product frontend
apps/api                 FastAPI product API entrypoint
packages/relia_core      Product-facing wrapper for reliance-state scoring
src/                     Python ML, data, evaluation, rules, and API source
infra/                   Docker Compose and deployment scaffolding
docs/                    Architecture, model cards, portfolio and deployment docs
tests/                   Unit, integration, policy/statistics, and smoke tests
artifacts/               Ignored generated data/model outputs
paper/                   Ignored manuscript and submission artifacts
.private/                Ignored local-only legacy/reviewer materials
```

## Core Features

- **Interactive case simulator:** enter initial answer, confidence, AI advice, final answer, ground truth, and context.
- **Reliance-state prediction:** beneficial AI reliance, harmful overreliance, harmful underreliance, correct self-reliance, independent correctness/error, or uncertain disagreement.
- **Rule-trace explanations:** human-readable active rules and counterfactual suggestions.
- **Conformal-gating dashboard:** alpha, threshold, harmful-case capture, missed harmful cases, intervention burden, and non-intervention coverage.
- **Model-evaluation lab:** AUROC, AUPRC, F1, balanced accuracy, Brier score, ECE, calibration summaries, and cross-dataset transfer.
- **Policy simulation:** compare no gating, confidence thresholds, symbolic-rule gating, and ReliaGuard-NS conformal gating.
- **MLOps hooks:** optional MLflow experiment logging and model-card documentation.
- **Safety boundaries:** clear limitations for non-causal, non-clinical, non-diagnostic AI evaluation.

## Quickstart

One-command local app launcher on Windows:

```powershell
.\run_app.ps1
```

Or from Command Prompt / double-click:

```bat
run_app.bat
```

Backend:

```powershell
python -m pip install -e ".[dev]"
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd apps/web
npm install
npm run dev
```

Full stack:

```powershell
docker compose -f infra/docker-compose.yml up --build
```

## Publishing As A New GitHub Repository

This project should be published as a new repository named `ReliaGuard-Studio`, not as the old NeuroSymbolic prototype repository. See [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md) for the exact first-push commands and privacy checks.

## API Examples

```powershell
Invoke-RestMethod http://127.0.0.1:8000/model-card
Invoke-RestMethod http://127.0.0.1:8000/datasets
Invoke-RestMethod "http://127.0.0.1:8000/conformal-threshold?alpha=0.10"
```

Reliance prediction:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/predict-reliance `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"initial_answer":"A","initial_confidence":0.82,"ai_advice":"B","final_answer":"B","ground_truth":"A","task_context":"loan review","advice_source":"AI","model_confidence":0.78}'
```

## Reproduce the Public Analysis Pipeline

These commands download/prepare public data into ignored `artifacts/` folders and regenerate the public analysis artifacts.

```powershell
nsca search-datasets
nsca download-real-data
nsca prepare-real-data
nsca run-real-experiments
nsca run-cross-dataset
nsca run-policy-evaluation
nsca run-conformal-risk-control
nsca run-off-policy-evaluation
nsca run-sensitivity-analyses
nsca run-negative-controls
nsca run-leakage-audit
pytest
python -m compileall src
```

## Evidence Boundaries

Supported:

- Decision reliance, overreliance, underreliance, correct self-reliance, confidence movement, and outcome changes where dataset fields support them.
- Short-term tutoring learning-gain analysis for the public tutoring dataset.
- Observational GenAI process-trace analysis for FLoRA.
- Calibration, cross-dataset transfer limits, rule ablations, and non-causal policy simulation.

Not supported:

- Clinical or diagnostic claims.
- Cognitive decline claims.
- Delayed recall, transfer, or long-term learning claims.
- Claims that showing a ReliaGuard warning causally changes deployed human behavior without a prospective randomized validation study.

## Project Summary and Portfolio Material

- Project summary PDF: `docs/project_summary/project_summary.pdf`
- Architecture docs: `docs/architecture.md`
- Model card: `docs/model_cards/reliaguard_ns.md`
- Portfolio and resume copy: `docs/portfolio/README.md`
- Deployment guide: `docs/deployment.md`

## Publication Safety

This public repository is intentionally product/code-first. Manuscript source, generated paper figures, raw downloaded datasets, prepared participant-level data, reviewer materials, and private audits are ignored so they are not accidentally published.

Before pushing to GitHub:

```powershell
git status --ignored
git check-ignore -v paper artifacts .private Docs
```

Choose a repository license and update `CITATION.cff` before public release.
