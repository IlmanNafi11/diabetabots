"""Shared pytest fixtures for the DiabetaBot test suite."""

import os
from collections.abc import AsyncGenerator, Iterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings
from app.database import create_async_engine, create_sessionmaker
from app.main import create_app
from app.models.models import Base

# Test database URL. Defaults to a local PostgreSQL instance matching the
# compose credentials; override with TEST_DATABASE_URL in CI.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://diabetabot:diabetabot@localhost:5432/diabetabot_test",
)


def make_test_settings(**overrides: object) -> Settings:
    """Build a Settings instance valid for tests (credentials + database)."""
    base: dict[str, object] = {
        "database_url": TEST_DATABASE_URL,
        "telegram_bot_token": "test-token-123",
        "llm_api_key": "sk-test-key",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def app() -> FastAPI:
    """Return a fresh application instance per test."""
    return create_app()


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a test client wired to the application under test."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def test_settings() -> Settings:
    """Return settings pointed at the test database, with valid credentials."""
    return Settings(
        database_url=TEST_DATABASE_URL,
        telegram_bot_token="test-token-123",
        llm_api_key="sk-test-key",
    )


@pytest_asyncio.fixture()
async def async_engine(test_settings: Settings) -> AsyncGenerator[AsyncEngine, None]:
    """Provide an engine with a clean schema; drop it afterwards."""
    engine = create_async_engine(test_settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest_asyncio.fixture()
async def sessionmaker(async_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to the isolated test engine."""
    return create_sessionmaker(async_engine)


@pytest_asyncio.fixture()
async def async_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a session for tests that need to touch the test database."""
    async with sessionmaker() as session:
        yield session


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------
# A skipped test reports success to the exit code, which turns a CI quality
# gate into a false green. When FAIL_ON_SKIP=1 (set by CI), any test that
# skips for a reason other than an explicit ``requires_ollama`` deselection
# fails the session instead.
_unexpected_skips: list[str] = []


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record skips so the session can fail on them at the end."""
    if report.skipped and report.when in ("setup", "call"):
        _unexpected_skips.append(f"{report.nodeid}: {report.longrepr}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run when skips appeared under FAIL_ON_SKIP=1."""
    if os.environ.get("FAIL_ON_SKIP") != "1" or not _unexpected_skips:
        return
    session.exitstatus = 1
    print(f"\nFAIL_ON_SKIP: {len(_unexpected_skips)} unexpected skipped test(s):")
    for entry in _unexpected_skips:
        print(f"  - {entry}")
