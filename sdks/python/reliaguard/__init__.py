from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


def _camel_to_snake(payload: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "projectId": "project_id",
        "userId": "user_id",
        "taskId": "task_id",
        "initialAnswer": "initial_answer",
        "initialConfidence": "initial_confidence",
        "aiAdvice": "ai_advice",
        "aiConfidence": "ai_confidence",
        "finalAnswer": "final_answer",
        "groundTruth": "ground_truth",
    }
    return {mapping.get(key, key): value for key, value in payload.items()}


@dataclass
class ReliaGuard:
    """Tiny ReliaGuard Studio SDK.

    The SDK defaults to the local development server. In production, point
    ``base_url`` at the deployed FastAPI service and pass an API key once auth
    is enabled.
    """

    api_key: str = "local-dev"
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 10.0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def log_interaction(self, **payload: Any) -> dict[str, Any]:
        """Stream a completed human-AI interaction for audit/shadow monitoring."""

        response = httpx.post(
            f"{self.base_url}/v1/events/log",
            json=_camel_to_snake(payload),
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def check_guardrail(self, **payload: Any) -> dict[str, Any]:
        """Call ReliaGuard during a decision and get an intervention action."""

        response = httpx.post(
            f"{self.base_url}/v1/guardrail/check",
            json=_camel_to_snake(payload),
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


__all__ = ["ReliaGuard"]
