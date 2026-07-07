from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check_capabilities():
    response = client.get("/health")
    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert "image_generate" in capabilities
    assert "image_recognize" in capabilities

def test_generate_image_basic():
    response = client.post("/image/generate", json={"prompt": "A futuristic city"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "image_url" in data
    assert data["used_references"] == []

def test_generate_image_with_character():
    response = client.post("/image/generate", json={"prompt": "Jessie at the park", "character_name": "jessie_crider"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["used_references"]) > 0
    assert "jessie-crider-reference-1.jpg" in data["used_references"][0]

def test_generate_image_invalid_character():
    response = client.post("/image/generate", json={"prompt": "Unknown person", "character_name": "non_existent"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["used_references"] == []

def test_analyze_image():
    response = client.post("/image/analyze", json={"image_url": "example_image_url"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "analysis" in data
    assert "detected_objects" in data
