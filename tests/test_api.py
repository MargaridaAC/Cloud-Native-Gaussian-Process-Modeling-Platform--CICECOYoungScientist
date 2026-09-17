import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402
from server import app  # noqa: E402

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_session_isolation():
    session1 = "session-user-alpha"
    session2 = "session-user-beta"

    # Train model on session 1
    resp1 = client.post(
        "/api/train-manual",
        headers={"X-Session-ID": session1},
        json={
            "x_points": [1.0, 2.0, 3.0, 4.0],
            "y_points": [2.0, 4.0, 6.0, 8.0],
            "kernel": "RBF",
        },
    )
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "success"

    # Check model info on session 1: train_done should be True
    info1 = client.get("/api/model-info", headers={"X-Session-ID": session1})
    assert info1.status_code == 200
    assert info1.json()["train_done"] is True

    # Check model info on session 2: train_done should be False
    info2 = client.get("/api/model-info", headers={"X-Session-ID": session2})
    assert info2.status_code == 200
    assert info2.json()["train_done"] is False


def test_train_and_predict():
    session_id = "session-train-predict-test"

    # 1. Train manual 1D model
    train_resp = client.post(
        "/api/train-manual",
        headers={"X-Session-ID": session_id},
        json={
            "x_points": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_points": [1.5, 3.1, 4.8, 6.9, 9.2],
            "kernel": "RBF",
            "norm_label": "None",
            "norm_feat": "None",
            "likelihood": False,
            "white_kernel": False,
        },
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["status"] == "success"
    assert train_data["n_train"] == 5

    # 2. Predict Y for a target point
    pred_resp = client.post(
        "/api/predict-y",
        headers={"X-Session-ID": session_id},
        json={"x_values": [2.5], "confidence_level": 95.0},
    )
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert "pred_y" in pred_data
    assert "std_y" in pred_data
    assert "ci_lower" in pred_data
    assert "ci_upper" in pred_data
    assert pred_data["ci_lower"] <= pred_data["pred_y"] <= pred_data["ci_upper"]
