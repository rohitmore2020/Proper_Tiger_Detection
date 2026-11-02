from fastapi.testclient import TestClient
from pathlib import Path
import pytest

from app.main import app

client = TestClient(app)


def test_database_and_history_workflow():
    test_img = Path("dataset/test/000095.jpg")
    if not test_img.exists():
        test_img = Path("../dataset/test/000095.jpg")

    assert test_img.exists()

    # Trigger detection which writes to SQLite DB
    with open(test_img, "rb") as f:
        response = client.post(
            "/api/v1/detect-and-identify",
            files={"file": ("000095.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    res_data = response.json()
    record_id = res_data["id"]
    assert record_id is not None

    # Fetch history list
    hist_resp = client.get("/api/v1/history?limit=10")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["total"] >= 1
    assert any(r["id"] == record_id for r in hist_data["records"])

    # Fetch single record
    rec_resp = client.get(f"/api/v1/history/{record_id}")
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert rec_data["id"] == record_id
    assert rec_data["predicted_id"] == res_data["identification"]["matched_id"]
