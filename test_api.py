
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_chatbot_valid_question():
    response = client.post(
        "/assistant/chat",
        json={"question": "What PPE should workers use?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()


def test_chatbot_rejects_empty_question():
    response = client.post(
        "/assistant/chat",
        json={"question": ""}
    )
    assert response.status_code == 422


def test_daily_site_assistant():
    response = client.post(
        "/assistant/daily-site",
        json={
            "project_type": "Residential building",
            "workers": 15,
            "work_planned": "Brickwork on the first floor",
            "weather": "sunny",
            "issues": "None reported"
        }
    )
    assert response.status_code == 200
    assert "checklist" in response.json()


def test_weather_advisory():
    response = client.post(
        "/assistant/weather-advisory",
        json={
            "temperature_c": 38,
            "rain_expected": False,
            "wind_kmh": 10,
            "lightning_expected": False,
            "work_type": "general"
        }
    )
    assert response.status_code == 200
    assert "recommendations" in response.json()


def test_weather_rejects_invalid_temperature():
    response = client.post(
        "/assistant/weather-advisory",
        json={
            "temperature_c": 100,
            "rain_expected": False,
            "wind_kmh": 10,
            "work_type": "general"
        }
    )
    assert response.status_code == 422
