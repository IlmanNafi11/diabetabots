"""Tests for the health endpoint backing the compose healthcheck."""

import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app, create_production_app


def _clear_allowed_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop any ALLOWED_HOSTS inherited from the developer's shell or .env."""
    for key in list(os.environ):
        if key.upper() == "ALLOWED_HOSTS":
            monkeypatch.delenv(key, raising=False)


def _production_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "real-token")
    monkeypatch.setenv("LLM_API_KEY", "real-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/d")


class _NoopRuntime:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host-header assertions free of database and embedding clients."""
    monkeypatch.setattr("app.runtime.build_runtime", lambda *_, **__: _NoopRuntime())


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


def test_production_app_accepts_local_healthcheck_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The image healthcheck probes 127.0.0.1, so it must pass with no tuning."""
    _clear_allowed_hosts(monkeypatch)
    _production_env(monkeypatch)
    _stub_runtime(monkeypatch)

    with TestClient(create_production_app()) as client:
        response = client.get("/health", headers={"host": "127.0.0.1:8000"})

    assert response.status_code == 200


def test_production_app_reads_allowed_hosts_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployer adds its public hostname without rebuilding the image."""
    _clear_allowed_hosts(monkeypatch)
    _production_env(monkeypatch)
    _stub_runtime(monkeypatch)
    monkeypatch.setenv("ALLOWED_HOSTS", "bot.example.com")

    with TestClient(create_production_app()) as client:
        allowed = client.get("/health", headers={"host": "bot.example.com"})
        rejected = client.get("/health", headers={"host": "evil.example.com"})

    assert allowed.status_code == 200
    assert rejected.status_code == 400
