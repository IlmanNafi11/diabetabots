"""Tests for the health endpoint backing the compose healthcheck."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_ok_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_route_is_not_served() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 404
