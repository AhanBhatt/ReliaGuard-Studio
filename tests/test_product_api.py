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
