# ReliaGuard Studio Architecture

ReliaGuard Studio is organized as a production-style AI evaluation platform.

```text
apps/web        Next.js + TypeScript product UI
apps/api        FastAPI service entrypoint
packages/relia_core
                Product-facing wrapper for the Python reliance engine
src/            Research and ML implementation source of truth
infra/          Docker Compose and deployment scaffolding
docs/           Architecture, model cards, portfolio/resume materials
artifacts/      Ignored raw data, prepared data, model outputs and metrics
paper/          Ignored manuscript/submission materials
```

## Runtime flow

1. A user enters an initial judgment, confidence, AI advice, final judgment, ground truth and context in the simulator.
2. The web app calls `POST /predict-reliance`.
3. The FastAPI service computes observable reliance-state labels, calibrated harmful-reliance risk, active rules, uncertainty and recommended action.
4. Dashboard endpoints serve artifact-backed model metrics, conformal thresholds, policy simulations and dataset summaries.

## Product boundary

ReliaGuard Studio is for AI evaluation, model-risk analysis and prospective intervention design. It is not a clinical, diagnostic, cognitive-decline, or causal deployment-effect system.
