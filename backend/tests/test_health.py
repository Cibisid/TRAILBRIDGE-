"""
TrialBridge — Test Suite
Day 1 test: verify the health endpoint returns 200.
More tests added each day as features are built.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


@pytest.fixture
def client():
    """Provide a test client for the FastAPI app."""
    # We patch init_db and close_db so tests don't need a real database
    with patch("backend.main.init_db", new_callable=AsyncMock), \
         patch("backend.main.close_db", new_callable=AsyncMock):
        from backend.main import app
        with TestClient(app) as c:
            yield c


def test_health_check(client):
    """Health endpoint should return 200 with status: healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "TrialBridge"
    assert "version" in data
    assert "uptime_seconds" in data


def test_api_docs_available_in_dev(client):
    """OpenAPI docs should be accessible in development mode."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_stub_endpoints_return_200(client):
    """All stub endpoints should be reachable (return non-500)."""
    endpoints = [
        ("GET", "/api/v1/trials"),
        ("POST", "/api/v1/auth/token"),
        ("POST", "/api/v1/parse-patient"),
        ("POST", "/api/v1/match"),
    ]
    for method, path in endpoints:
        response = client.request(method, path)
        assert response.status_code != 500, f"{method} {path} returned 500"
