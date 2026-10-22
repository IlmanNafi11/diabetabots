"""Application configuration loaded from the environment.

All external parameters (Telegram token, LLM/embedding endpoints, database
URL, RAG and app settings) are declared here as typed, validated settings so
that business logic never reads ``os.environ`` directly (SPEC section 5.1).
"""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, sourced from environment variables.

    A ``.env`` file is loaded when present; real environment variables always
    take precedence. Field names map to upper-case env vars (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Telegram ---
    telegram_bot_token: str = "your-telegram-bot-token"

    # --- LLM provider (OpenAI-compatible) + self-hosted embedding (Ollama) ---
    llm_base_url: str = "https://your-llm-endpoint/v1"
    llm_api_key: str = "your-api-key"
    chat_model_name: str = "your-chat-model"
    ollama_base_url: str = "http://ollama:11434"
    # A single interactive query (one document) should answer fast: when the
    # endpoint stalls, the sync embed call must fail quickly so the Telegram
    # handler's worker thread is not pinned and the container healthcheck
    # stays responsive.
    ollama_query_read_timeout_seconds: float = Field(default=30.0, gt=0)
    # Embedding inference is CPU-bound; a full 100-document batch can take
    # minutes on slower hosts, so the batch (indexing/migration) read timeout
    # must exceed that.
    ollama_batch_read_timeout_seconds: float = Field(default=600.0, gt=0)
    embedding_model_name: str = "snowflake-arctic-embed2:latest"
    embedding_dim: int = Field(default=1024, gt=0)
    rag_top_k: int = Field(default=5, gt=0)
    # Retrieval candidate pool used to ground results against TKPI food names.
    # The service embeds the query, pulls this many nearest neighbours, then
    # ranks them by query/name token overlap so the correct entry is chosen even
    # when the embedding distance order is misleading. Must be >= rag_top_k.
    rag_candidate_k: int = Field(default=20, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- RAG config (winning chunking + embedding decided by the Iterasi 0 spike) ---
    chunking_strategy: str = "recursive"
    collection_name: str = "tkpi_gi"
    chroma_persist_dir: str = "./chroma_db"

    # --- Database (PostgreSQL, async) ---
    database_url: str = "postgresql+asyncpg://diabetabot:diabetabot@postgres:5432/diabetabot"

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, gt=0, le=65535)
    timezone: str = "Asia/Jakarta"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("ollama_base_url")
    @classmethod
    def _require_ollama_url(cls, value: str) -> str:
        """Fail fast on a malformed Ollama endpoint (e.g. missing scheme)."""
        if not (
            value.startswith("http://") or value.startswith("https://")
        ) or len(value) <= len("http://"):
            raise ValueError("ollama_base_url must be an http(s):// URL with a host")
        return value

    @field_validator("database_url")
    @classmethod
    def _require_asyncpg_driver(cls, value: str) -> str:
        """The async SQLAlchemy engine requires the asyncpg driver."""
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("database_url must use the async 'postgresql+asyncpg://' driver")
        return value
