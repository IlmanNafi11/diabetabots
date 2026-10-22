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
    # Building the runtime with the ``"your-*"`` placeholder aborts start-up
    # (credential hygiene: the bot would otherwise authenticate to Telegram
    # with a bogus token and log 401 storms).
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

    # Maximum characters accepted in a single user question before the handler
    # rejects it (protects against prompt-stuffing / token-exhaustion attacks).
    max_user_question_chars: int = Field(default=512, gt=0)

    # --- RAG config (winning chunking + embedding decided by the Iterasi 0 spike) ---
    chunking_strategy: str = "recursive"
    collection_name: str = "tkpi_gi"
    chroma_persist_dir: str = "./chroma_db"

    # --- Database (PostgreSQL, async) ---
    # Required — no default. Compose mode A passes DATABASE_URL through the
    # POSTGRES_* env vars; mode B (external host) binds it via .env. A missing
    # value fails fast at Settings construction instead of connecting with a
    # guessable default password.
    database_url: str

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, gt=0, le=65535)
    timezone: str = "Asia/Jakarta"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("database_url")
    @classmethod
    def _require_asyncpg_driver(cls, value: str) -> str:
        """The async SQLAlchemy engine requires the asyncpg driver."""
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("database_url must use the async 'postgresql+asyncpg://' driver")
        return value

    @field_validator("telegram_bot_token")
    @classmethod
    def _reject_sentinel_telegram_token(cls, value: str) -> str:
        if value.startswith("your-") or value.strip() == "":
            raise ValueError(
                "telegram_bot_token is still the sentinel placeholder. "
                "Set TELEGRAM_BOT_TOKEN in your .env before starting."
            )
        return value

    @field_validator("llm_api_key")
    @classmethod
    def _reject_sentinel_api_key(cls, value: str) -> str:
        if value.startswith("your-") or value.strip() == "":
            raise ValueError(
                "llm_api_key is still the sentinel placeholder. "
                "Set LLM_API_KEY in your .env before starting."
            )
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def _allowlist_ollama_base_url(cls, value: str) -> str:
        """SSRF guard for the self-hosted Ollama base URL.

        Only the compose service name or loopback hosts are permitted; any
        other host (public FQDN, raw IP, internal name) is rejected. See
        SECURITY.md "Ollama base URL allowlist".
        """
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError("ollama_base_url must be an http(s):// URL")
        host = value.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        allowlist = {"localhost", "127.0.0.1", "ollama"}
        if host not in allowlist:
            raise ValueError(
                "ollama_base_url host must be one of ollama/localhost/127.0.0.1 "
                "(SSRF allowlist; see SECURITY.md)."
            )
        return value

    @field_validator("max_user_question_chars")
    @classmethod
    def _require_positive_prompt_cap(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_user_question_chars must be positive")
        return value
