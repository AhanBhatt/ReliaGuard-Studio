# ReliaGuard Studio Usage Guide

ReliaGuard Studio is a production-style AI evaluation platform for detecting overreliance, underreliance, and unsafe human-AI decision behavior. It is designed to show how average AI-assisted accuracy can hide harmful reliance failures, then expose those failures through calibrated risk, symbolic rule traces, counterfactual explanations, and selective-gating analyses.

This guide explains how to run the project, use the web app, call the API, reproduce the analysis pipeline, and interpret the outputs safely.

## 1. What The Project Does

ReliaGuard Studio accepts a human-AI decision tuple:

- initial human answer;
- initial confidence;
- AI or support-system advice;
- final human answer;
- ground truth;
- task context;
- advice source;
- model confidence.

It returns:

- a reliance state such as beneficial AI reliance, harmful overreliance, harmful underreliance, correct self-reliance, independent correctness, independent error, or uncertain disagreement;
- harmful-reliance risk;
- uncertainty;
- active symbolic rules;
- a plain-language explanation;
- a counterfactual suggestion;
- a recommended selective-gating action.

The system is for AI evaluation and model-risk analysis. It is not a clinical tool, cognitive-decline system, or causal proof that a warning will change user behavior.

## 2. Repository Layout

```text
apps/web                 Next.js + TypeScript product frontend
apps/api                 FastAPI product API entrypoint
packages/relia_core      Small product-facing wrapper for reliance scoring
src/                     Python package with data, models, rules, evaluation and API source
infra/                   Docker Compose and deployment notes
docs/                    Architecture, model card, deployment, portfolio and usage docs
tests/                   Unit, API, model, policy, statistics and smoke tests
artifacts/               Ignored generated data/model outputs
paper/                   Ignored manuscript and paper artifacts
.private/                Ignored local-only research/reviewer materials
```

## 3. Install And Run Locally

Fastest option on Windows:

```powershell
.\run_app.ps1
```

This checks dependencies, starts the FastAPI backend and Next.js frontend, prints both local URLs, streams logs, and stops both processes when you press `Ctrl+C`. It does not open a browser automatically.

You can also run it from Command Prompt or by double-clicking:

```bat
run_app.bat
```

Optional launcher flags:

```powershell
.\run_app.ps1 -ApiPort 8010 -WebPort 3010
.\run_app.ps1 -SkipInstall
```

Manual setup:

From the repository root:

```powershell
python -m pip install -e ".[dev]"
```

Start the API:

```powershell
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

Start the web app in another terminal:

```powershell
cd apps/web
npm install
npm run dev
```

Open the app at:

```text
http://127.0.0.1:3000
```

The frontend calls the backend configured by:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## 4. Run With Docker Compose

```powershell
docker compose -f infra/docker-compose.yml up --build
```

The API runs on port `8000`; the web app runs on port `3000`.

## 5. Homepage Walkthrough

The homepage is a 30-second product pitch:

- the core thesis: average accuracy can hide unsafe reliance behavior;
- the five public datasets behind the analysis;
- the reliance-state output layer;
- the product actions: calibrated risk, rule trace, counterfactual suggestion and gating recommendation.

Use the homepage first if you are showing the project to a recruiter, professor, teammate, or reviewer. It explains the platform without requiring them to read the manuscript or source code.

## 6. Interactive Case Simulator

Path:

```text
/simulator
```

Use this page to score one human-AI decision.

Example harmful overreliance case:

- initial answer: `A`;
- AI advice: `B`;
- final answer: `B`;
- ground truth: `A`;
- initial confidence: `0.82`;
- model confidence: `0.78`;
- task context: `loan decision support`.

This should return a harmful overreliance state because the user was initially correct, the advice was wrong, and the final answer moved toward the wrong advice.

Read the output in this order:

1. Predicted state: the main reliance-state classification.
2. Harmful reliance risk: a calibrated-style risk signal for selective intervention.
3. Uncertainty: how ambiguous the case is given the available signals.
4. Explanation card: the human-readable reason for the classification.
5. Active rules: symbolic rule families that fired.
6. Counterfactual: what would reduce risk in a similar case.
7. Recommended action: the product-level gating suggestion.

## 7. Conformal Gating Dashboard

Path:

```text
/gating
```

This page lets you move an alpha slider and inspect the trade-off between harmful-case capture and intervention burden.

Interpretation:

- lower alpha means stricter harmful-case capture;
- stricter capture usually increases intervention burden;
- higher alpha allows more missed harmful cases and usually lowers burden.

Important detail: if exact calibrated artifacts exist only for one alpha, the dashboard shows a clearly labelled preview estimate for other alpha values. This keeps the interface interactive while preserving the evidence boundary. For scientific reporting or deployment review, regenerate exact conformal artifacts at the desired alpha:

```powershell
nsca run-conformal-risk-control
```

The dashboard reports:

- threshold;
- harmful-case capture;
- missed harmful fraction;
- intervention burden;
- non-intervention coverage where available;
- whether the row is an exact artifact or a preview estimate.

## 8. Model Evaluation Lab

Path:

```text
/evaluation
```

This page compares model outputs across datasets, targets and validation splits. It is meant to show that the project evaluates models properly instead of only training them.

Metrics include:

- AUROC;
- AUPRC;
- Brier score;
- expected calibration error;
- calibration slope/intercept when available;
- split type;
- dataset and target.

ReliaGuard-NS should not be described as a universal raw-prediction winner. The intended product value is calibrated, interpretable selective-risk diagnosis.

## 9. Policy Simulation

Path:

```text
/policy
```

This page compares candidate gating strategies:

- no gating;
- confidence-threshold gating;
- symbolic-rule gating;
- ReliaGuard-NS conformal gating.

Use this page to understand planning trade-offs:

- final correctness;
- overreliance;
- underreliance;
- intervention burden;
- utility proxy when available.

The policy simulation is observational. It can help design a prospective validation study, but it does not prove that showing an intervention will causally change behavior.

## 10. Data And Reproducibility Page

Path:

```text
/data
```

This page explains the public dataset pipeline and gives the main commands:

```powershell
python -m pip install -e ".[dev]"
nsca download-real-data
nsca prepare-real-data
nsca run-real-experiments
nsca run-cross-dataset
nsca run-policy-evaluation
nsca run-conformal-risk-control
```

Generated outputs go into ignored artifact folders so the GitHub repository does not accidentally publish raw participant data or unpublished paper artifacts.

## 11. API Usage

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Dataset summary:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/datasets
```

Model card:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/model-card
```

Conformal threshold:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/conformal-threshold?alpha=0.20"
```

Reliance prediction:

```powershell
$body = @{
  initial_answer = "A"
  initial_confidence = 0.82
  ai_advice = "B"
  final_answer = "B"
  ground_truth = "A"
  task_context = "loan decision support"
  advice_source = "AI"
  model_confidence = 0.78
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/predict-reliance `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Plain-language explanation:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/explain-case `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 12. Development Checks

Run Python checks:

```powershell
python -m compileall src tests
pytest
```

Run frontend checks:

```powershell
cd apps/web
npm run typecheck
npm run build
```

## 13. What To Say In A Portfolio Or Interview

Short version:

> ReliaGuard Studio is a full-stack AI evaluation platform that detects when AI assistance improves average accuracy while hiding unsafe human reliance failures. It integrates five public datasets and exposes calibrated risk, symbolic explanations, conformal thresholds and policy simulations through a FastAPI + Next.js product interface.

Resume-style bullets:

- Built a full-stack AI evaluation platform using FastAPI, Next.js, TypeScript and Tailwind to detect overreliance, underreliance and unsafe human-AI decision behavior.
- Integrated a neuro-symbolic reliance-state engine with calibrated risk, symbolic rule traces, counterfactual explanations and selective-gating policy outputs.
- Evaluated 43,263 public records across five human-AI datasets, including decision advice, XAI interfaces, tutoring support and GenAI process traces.
- Added reproducible ML evaluation workflows with calibration metrics, cross-dataset transfer, policy simulation, model cards, Docker Compose and CI-ready tests.

## 14. Safety And Limitation Boundaries

Do not claim:

- clinical diagnosis;
- cognitive decline detection;
- delayed recall or transfer learning unless supported by future integrated data;
- causal intervention effectiveness from the offline policy simulator;
- guaranteed prevention of harmful AI reliance.

Safe phrasing:

- “AI evaluation platform”;
- “selective-risk analysis”;
- “observational policy simulation”;
- “calibrated-style risk signal”;
- “prospective validation scaffold”;
- “not causal without a randomized intervention study.”
