# ReliaGuard Python SDK

Install locally:

```bash
python -m pip install -e sdks/python
```

Log an interaction:

```python
from reliaguard import ReliaGuard

rg = ReliaGuard(api_key="local-dev")

rg.log_interaction(
    project_id="support-copilot",
    user_id="agent_123",
    task_id="ticket_456",
    initial_answer="refund",
    initial_confidence=0.72,
    ai_advice="deny_refund",
    ai_confidence=0.88,
    final_answer="deny_refund",
    ground_truth="refund",
    context={"domain": "customer_support", "model": "gpt-4.1"},
)
```

Call the live guardrail:

```python
decision = rg.check_guardrail(
    project_id="support-copilot",
    user_id="agent_123",
    task_id="ticket_456",
    initial_answer="refund",
    initial_confidence=0.72,
    ai_advice="deny_refund",
    ai_confidence=0.88,
    final_answer="deny_refund",
    ground_truth="refund",
    context={"domain": "customer_support"},
)
print(decision["recommended_action"])
```
