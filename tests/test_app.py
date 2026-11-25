
from fastapi.testclient import TestClient

from app.main import app, model


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "NYC Taxi Prediction API is running"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict(monkeypatch):
    # Replace the real model prediction with a fixed test result
    def fake_predict(input_data):
        return [12.5]

    monkeypatch.setattr(model, "predict", fake_predict)

    response = client.post(
        "/predict",
        json={
            "pickup_datetime": "2025-01-15T14:30:00",
            "PULocationID": 161,
            "DOLocationID": 236,
            "passenger_count": 2
        }
    )

    assert response.status_code == 200
    assert response.json() == {"predicted_duration": 12.5}


def test_predict_invalid_location():
    response = client.post(
        "/predict",
        json={
            "pickup_datetime": "2025-01-15T14:30:00",
            "PULocationID": -1,
            "DOLocationID": 236,
            "passenger_count": 2
        }
    )

    assert response.status_code == 422
