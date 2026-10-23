"""Deployment contract tests for task-14."""

import subprocess
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
    # The production command now lives in the entrypoint: the default must
    # still boot the runtime factory, otherwise the Telegram polling loop
    # and reset scheduler never start (task-14).
    assert 'ENTRYPOINT ["docker-entrypoint"]' in dockerfile
    assert 'CMD ["serve"]' in dockerfile
    entrypoint = (ROOT / "docker-entrypoint.sh").read_text()
    assert "create_production_app" in entrypoint


def test_compose_mode_a_contains_app_postgres_and_persistent_volumes() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "services:" in compose
    assert "  app:" in compose
    assert "  postgres:" in compose
    assert "condition: service_healthy" in compose
    assert "chroma_data:/app/chroma_db" in compose
    assert "pg_data:/var/lib/postgresql/data" in compose
    assert "healthcheck:" in compose


def test_external_postgres_override_removes_local_dependency_and_service() -> None:
    override = (ROOT / "docker-compose.external-postgres.yml").read_text()
    assert "postgres: !reset null" in override
    assert "external: true" in override
    assert "${EXTERNAL_NETWORK:?EXTERNAL_NETWORK must be set}" in override


def test_compose_mode_a_includes_ollama_server_and_init_with_volume() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "  ollama:" in compose
    assert "  ollama-init:" in compose
    # The server keeps downloaded models in a named volume; the one-shot init
    # reuses the same volume so the pull happens exactly once.
    assert "ollama_data:/root/.ollama" in compose
    assert "snowflake-arctic-embed2" in compose
    assert "service_completed_successfully" in compose


def test_external_ollama_override_resets_services_and_requires_base_url() -> None:
    override = (ROOT / "docker-compose.external-ollama.yml").read_text()
    assert "ollama: !reset null" in override
    assert "ollama-init: !reset null" in override
    assert "${OLLAMA_BASE_URL:?OLLAMA_BASE_URL must be set}" in override


def test_docker_entrypoint_dispatches_index_command() -> None:
    dispatch = (ROOT / "docker-entrypoint.sh").read_text()

    assert "serve)" in dispatch
    assert "index)" in dispatch
    assert "index_data" in dispatch


def test_docker_entrypoint_rejects_unknown_command() -> None:
    result = subprocess.run(
        ["sh", str(ROOT / "docker-entrypoint.sh"), "no-such-command"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 64
    assert "no-such-command" in result.stderr


def test_dockerfile_installs_entrypoint_and_indexing_payload() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "docker-entrypoint.sh" in dockerfile
    assert "COPY data ./data" in dockerfile
    assert "COPY scripts ./scripts" in dockerfile


def test_env_example_documents_ollama_embedding_defaults() -> None:
    env = (ROOT / ".env.example").read_text()

    assert "OLLAMA_BASE_URL=http://ollama:11434" in env
    assert "OLLAMA_QUERY_READ_TIMEOUT_SECONDS=30" in env
    assert "OLLAMA_BATCH_READ_TIMEOUT_SECONDS=600" in env
    assert "EMBEDDING_MODEL_NAME=snowflake-arctic-embed2:latest" in env
    assert "EMBEDDING_DIM=1024" in env
