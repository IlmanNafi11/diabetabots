"""Tests for app/config.py — pydantic-settings configuration."""

import os

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-123")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@host:5432/db")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example.com:11434")
    monkeypatch.setenv("OLLAMA_QUERY_READ_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OLLAMA_BATCH_READ_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("CHAT_MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("RAG_TOP_K", "5")
    monkeypatch.setenv("RAG_CANDIDATE_K", "20")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("CHUNKING_STRATEGY", "recursive")
    monkeypatch.setenv("COLLECTION_NAME", "tkpi_gi")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./chroma_db")
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.setenv("APP_PORT", "8000")
    monkeypatch.setenv("TIMEZONE", "Asia/Jakarta")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    s = Settings()

    assert s.telegram_bot_token == "test-token-123"
    assert s.database_url == "postgresql+asyncpg://user:pass@host:5432/db"
    assert s.llm_base_url == "https://llm.example.com/v1"
    assert s.llm_api_key == "sk-test"
    assert s.chat_model_name == "gpt-4o"
    assert s.embedding_model_name == "text-embedding-3-small"
    assert s.embedding_dim == 1536
    assert s.rag_top_k == 5
    assert s.rag_candidate_k == 20
    assert s.llm_temperature == 0.0
    assert s.chunking_strategy == "recursive"
    assert s.collection_name == "tkpi_gi"
    assert s.chroma_persist_dir == "./chroma_db"
    assert s.app_host == "0.0.0.0"
    assert s.app_port == 8000
    assert s.timezone == "Asia/Jakarta"
    assert s.ollama_base_url == "http://ollama.example.com:11434"
    assert s.ollama_query_read_timeout_seconds == 12.5
    assert s.ollama_batch_read_timeout_seconds == 900.0
    assert s.log_level == "INFO"


def test_settings_rejects_non_asyncpg_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")

    with pytest.raises(ValidationError, match="database_url"):
        Settings()


def test_settings_rejects_invalid_embedding_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIM", "-1")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_invalid_rag_candidate_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CANDIDATE_K", "0")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_temperature_above_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TEMPERATURE", "1.5")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_invalid_app_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_PORT", "99999")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_invalid_ollama_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for bad in ("ftp://ollama:11434", "ollama:11434", "http://"):
        monkeypatch.setenv("OLLAMA_BASE_URL", bad)
        with pytest.raises(ValidationError):
            Settings()


def test_settings_rejects_invalid_ollama_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_QUERY_READ_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.setenv("OLLAMA_QUERY_READ_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OLLAMA_BATCH_READ_TIMEOUT_SECONDS", "-5")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify defaults match .env.example values."""
    # Clear all env vars so defaults apply
    for key in list(os.environ):
        if key.isupper():
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/d")

    s = Settings(_env_file=None)

    assert s.telegram_bot_token == "your-telegram-bot-token"
    assert s.llm_base_url == "https://your-llm-endpoint/v1"
    assert s.llm_api_key == "your-api-key"
    assert s.chat_model_name == "your-chat-model"
    assert s.ollama_base_url == "http://ollama:11434"
    assert s.embedding_model_name == "snowflake-arctic-embed2:latest"
    assert s.embedding_dim == 1024
    assert s.ollama_query_read_timeout_seconds == 30.0
    assert s.ollama_batch_read_timeout_seconds == 600.0
    assert s.rag_top_k == 5
    assert s.rag_candidate_k == 20
    assert s.llm_temperature == 0.0
    assert s.chunking_strategy == "recursive"
    assert s.collection_name == "tkpi_gi"
    assert s.chroma_persist_dir == "./chroma_db"
    assert s.app_host == "0.0.0.0"
    assert s.app_port == 8000
    assert s.timezone == "Asia/Jakarta"
    assert s.log_level == "INFO"
