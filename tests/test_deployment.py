"""Deployment contract tests for task-14."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app

ROOT = Path(__file__).parents[1]


class _RecordingRuntime:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


def test_app_lifespan_starts_and_stops_injected_runtime() -> None:
    runtime = _RecordingRuntime()

    with TestClient(create_app(runtime=runtime)):
        assert runtime.started is True
        assert runtime.stopped is False

    assert runtime.stopped is True


def test_dockerfile_runs_as_non_root_and_exposes_healthcheck() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "USER appuser" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert 'CMD ["uvicorn", "app.main:app"' in dockerfile


def test_compose_mode_a_contains_app_postgres_and_persistent_volumes() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "services:" in compose
    assert "  app:" in compose
    assert "  postgres:" in compose
    assert "condition: service_healthy" in compose
    assert "chroma_data:/app/chroma_db" in compose
    assert "pg_data:/var/lib/postgresql/data" in compose
    assert "healthcheck:" in compose


def test_external_postgres_override_removes_local_dependency_and_uses_external_network() -> None:
    override = (ROOT / "docker-compose.external-postgres.yml").read_text()

    assert "depends_on: {}" in override
    assert "external: true" in override
    assert "${EXTERNAL_NETWORK:?EXTERNAL_NETWORK must be set}" in override
