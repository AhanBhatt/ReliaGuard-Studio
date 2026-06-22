from __future__ import annotations

from fastapi.testclient import TestClient

from reliaguard_studio.api.main import app


def test_predict_reliance_endpoint_flags_overreliance():
    client = TestClient(app)
    payload = {
        "initial_answer": "A",
        "initial_confidence": 0.82,
        "ai_advice": "B",
        "final_answer": "B",
        "ground_truth": "A",
        "task_context": "loan review",
        "advice_source": "AI",
        "model_confidence": 0.78,
    }
    response = client.post("/predict-reliance", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "harmful_overreliance"
    assert "wrong_advice_overreliance_risk" in body["active_rules"]
    assert body["safety_boundary"].startswith("Evaluation support only")


def test_product_metadata_endpoints():
    client = TestClient(app)
    assert client.get("/model-card").status_code == 200
    assert client.get("/datasets").status_code == 200
    assert client.get("/conformal-threshold?alpha=0.10").status_code == 200


def test_conformal_threshold_slider_values_do_not_go_blank():
    client = TestClient(app)
    exact = client.get("/conformal-threshold?alpha=0.10")
    preview = client.get("/conformal-threshold?alpha=0.20")
    assert exact.status_code == 200
    assert preview.status_code == 200
    assert exact.json()["thresholds"]
    assert preview.json()["thresholds"]
    assert preview.json()["alpha"] == 0.20
    assert preview.json()["alpha_source"] in {"exact_artifact", "preview_estimate_from_nearest_artifact"}


def test_guardrail_endpoint_returns_actionable_response():
    client = TestClient(app)
    payload = {
        "project_id": "support-copilot",
        "user_id": "agent_123",
        "task_id": "ticket_456",
        "initial_answer": "refund",
        "initial_confidence": 0.72,
        "ai_advice": "deny_refund",
        "ai_confidence": 0.88,
        "final_answer": "deny_refund",
        "ground_truth": "refund",
        "context": {"domain": "customer_support", "model": "gpt-4.1"},
    }
    response = client.post("/v1/guardrail/check", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "harmful_overreliance"
    assert body["recommended_action"] == "request_verification"
    assert body["message"]
    assert "active_rules" in body


def test_guardrail_without_ground_truth_uses_proxy_uncertainty_boundary():
    client = TestClient(app)
    payload = {
        "project_id": "support-copilot",
        "user_id": "agent_123",
        "task_id": "live_ticket",
        "initial_answer": "refund",
        "initial_confidence": 0.72,
        "ai_advice": "deny_refund",
        "ai_confidence": 0.88,
        "context": {"domain": "customer_support"},
    }
    response = client.post("/v1/guardrail/check", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "uncertain_disagreement"
    assert "ground_truth_unavailable" in body["active_rules"]
    assert "proxy risk" in body["safety_boundary"]


def test_ingestion_audit_and_review_queue_workflow():
    client = TestClient(app)
    demo = client.get("/v1/demo/customer-support").json()
    response = client.post(
        "/v1/ingest/validate",
        json={
            "project_id": "pytest-support-copilot",
            "source_name": "pytest_demo",
            "records": demo["records"],
            "mapping": demo["mapping"],
            "file_type": "manual",
        },
    )
    assert response.status_code == 200
    audit = response.json()
    assert audit["scored_records"] == len(demo["records"])
    assert "reliance_state_audit" in audit["analyses_possible"]
    assert audit["highest_risk_cases"]

    queue = client.get("/v1/review-queue?project_id=pytest-support-copilot")
    assert queue.status_code == 200
    assert queue.json()["cases"]


def test_projects_interventions_monitoring_and_replay_endpoints():
    client = TestClient(app)
    assert client.get("/v1/projects").status_code == 200
    assert client.get("/v1/interventions").json()["interventions"]
    assert client.get("/v1/monitoring").status_code == 200
    replay = client.post("/v1/replay", json={"project_id": "pytest-support-copilot", "alpha": 0.1, "policy": "reliaguard_ns"})
    assert replay.status_code == 200
    assert "intervention_burden" in replay.json()
