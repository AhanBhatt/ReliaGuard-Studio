# Guardrail API

Use the guardrail endpoint during an AI-assisted decision when your product needs an action recommendation before a user finalizes.

## Endpoint

```http
POST /v1/guardrail/check
```

## Request

```json
{
  "project_id": "support-copilot",
  "user_id": "agent_123",
  "task_id": "ticket_456",
  "initial_answer": "refund",
  "initial_confidence": 0.72,
  "ai_advice": "deny_refund",
  "ai_confidence": 0.88,
  "final_answer": "deny_refund",
  "ground_truth": "refund",
  "context": {
    "domain": "customer_support",
    "model": "gpt-4.1"
  },
  "mode": "guardrail"
}
```

## Response

```json
{
  "state": "harmful_overreliance",
  "risk": 0.91,
  "uncertainty": 0.76,
  "recommended_action": "request_verification",
  "message": "Ask the user to compare evidence before accepting the AI recommendation.",
  "active_rules": [
    "human_advice_disagreement",
    "wrong_advice_overreliance_risk",
    "advice_adoption"
  ],
  "case_id": "a1b2c3d4e5f6",
  "mode": "guardrail",
  "intervention_template": "Before accepting this AI suggestion, compare it against one piece of evidence.",
  "safety_boundary": "Guardrail output is decision support and observability. It is not a causal or clinical claim."
}
```

## Recommended action values

- `allow`
- `request_verification`
- `show_uncertainty`
- `delay`
- `route_to_review`

## Deployment modes

- **Audit Mode:** upload historical logs and generate reports.
- **Shadow Mode:** score live events without intervening.
- **Guardrail Mode:** return intervention actions to the host AI product.

## Ground-truth boundary

If `ground_truth` is available, ReliaGuard can assign confirmed reliance states such as harmful overreliance or harmful underreliance.

If `ground_truth` is not available during a live decision, ReliaGuard returns a proxy risk state such as `uncertain_disagreement`, includes `ground_truth_unavailable` in `active_rules`, and recommends a reversible action such as `delay`, `show_uncertainty`, or `route_to_review`.
