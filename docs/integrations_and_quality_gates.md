# Integrations And Evaluation Regression Gates

ReliaGuard Studio is designed to sit next to AI engineering observability tools, not replace them.

## Export and integration targets

- **MLflow:** experiment tracking, model-card metadata, metrics comparison, and model-version tags. The repository includes `src/reliaguard_studio/evaluation/mlflow_tracking.py` as the integration seam.
- **LLM observability tools:** ReliaGuard events can be attached to Langfuse-style traces as evaluation observations: state, risk, uncertainty, active rules, and recommended action.
- **OpenTelemetry-style GenAI telemetry:** the canonical event schema intentionally separates user, task, initial answer, advice, final answer, model confidence, ground truth, and context so it can be serialized as trace attributes.

## Regression gates to add to CI for production teams

The existing test suite covers API, rules, schema, metrics, leakage, and product endpoints. For a deployment, add thresholded model-quality gates:

- fail if AUROC drops by more than 3 percentage points on a locked benchmark;
- fail if ECE worsens by more than 5 percentage points;
- fail if conformal harmful-case capture falls below expected tolerance;
- fail if API response schemas change without a version bump;
- fail if leakage checks detect participant/task overlap in strict splits;
- fail if rule explanations become empty for known reliance scenarios.

## Suggested deployment stages

1. **Audit Mode:** upload historical logs and build a baseline risk profile.
2. **Shadow Mode:** stream live events without user-facing interventions.
3. **Guardrail Mode:** turn on recommended actions after reviewer calibration and burden analysis.

## Boundary

Offline replay and policy simulation estimate which cases would have been flagged. They do not prove that a warning would causally improve user behavior. Prospective randomized validation is required before making causal intervention claims.
