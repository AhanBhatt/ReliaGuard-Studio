from __future__ import annotations

from fastapi.testclient import TestClient

from reliaguard_studio.api.main import app


def test_api_endpoints_smoke() -> None:
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    config = client.get("/api/config")
    assert config.status_code == 200
    assert "task_families" in config.json()

    dashboard = client.get("/api/dashboard/summary")
    assert dashboard.status_code == 200
    assert dashboard.json()["summary"]["synthetic_validation_only"] is True

    sample = client.get("/api/demo/sample-session", params={"task_family": "coding"})
    assert sample.status_code == 200
    session_id = sample.json()["session"]["session_id"]

    scored = client.post("/api/score", json={"session_id": session_id})
    assert scored.status_code == 200
    report = scored.json()
    assert "fusion_overreliance_probability" in report
    assert "symbolic" in report
