"""
Tests for the FastAPI server
"""
from fastapi.testclient import TestClient

from server import app


def test_train():
    """training endpoint works with complete request object"""
    client = TestClient(app)
    response = client.post(
        "/train",
        json={
            "regularization_strength": 1.0,
            "training_size": 0.8,
            "target": "species",
        },
    )
    assert response.status_code == 200


def test_predict():
    """prediction returns a penguin species"""
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "flipper_length_mm": 250,
            "bill_length_mm": 45,
            "bill_depth_mm": 5,
            "sex": "Female",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["species"] == "Gentoo"


def test_predict_error():
    """wrong object returns in a validation error"""
    client = TestClient(app)
    response = client.post("/predict", json={"name": "Pingu"})
    assert response.status_code == 422  # BAD REQUEST
