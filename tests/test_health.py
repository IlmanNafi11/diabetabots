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


def test_production_http_surface_disables_interactive_docs() -> None:
    with TestClient(create_app(production=True)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_production_http_surface_rejects_untrusted_host() -> None:
    with TestClient(create_app(production=True, allowed_hosts=["bot.example.com"])) as client:
        response = client.get("/health", headers={"host": "evil.example.com"})

    assert response.status_code == 400


def test_production_http_surface_accepts_configured_host() -> None:
    with TestClient(create_app(production=True, allowed_hosts=["bot.example.com"])) as client:
        response = client.get("/health", headers={"host": "bot.example.com"})

    assert response.status_code == 200
