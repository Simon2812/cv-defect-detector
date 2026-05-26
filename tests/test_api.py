"""API smoke tests for the defect detection service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from defect_detector.api.main import app


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
