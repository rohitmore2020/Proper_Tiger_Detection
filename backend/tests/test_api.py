from fastapi.testclient import TestClient
from pathlib import Path
import pytest

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validate_image_endpoint():
    test_img = Path("dataset/test/000085.jpg")
    if not test_img.exists():
        test_img = Path("../dataset/test/000085.jpg")

    with open(test_img, "rb") as f:
        response = client.post(
            "/api/v1/images/validate",
            files={"file": ("000085.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "valid"


def test_detect_and_identify_endpoint():
    test_img = Path("dataset/test/000095.jpg")
    if not test_img.exists():
        test_img = Path("../dataset/test/000095.jpg")

    with open(test_img, "rb") as f:
        response = client.post(
            "/api/v1/detect-and-identify",
            files={"file": ("000095.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "detection" in data
    assert "identification" in data
    assert data["detection"]["detected"] is True
    assert len(data["detection"]["box"]) == 4
    assert data["identification"]["matched_id"] != ""
    assert data["processing_time_ms"] > 0
