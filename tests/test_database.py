"""Tests for app/database.py — async engine, session factory, get_db dependency."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings
from app.database import create_async_engine, get_db, init_db


@pytest.fixture()
def settings() -> Settings:
    """Use the shared test database URL so CI/local runs stay configurable."""
    from tests.conftest import TEST_DATABASE_URL

    return Settings(database_url=TEST_DATABASE_URL)


@pytest.mark.asyncio
async def test_create_async_engine_returns_async_engine(settings: Settings) -> None:
    engine = create_async_engine(settings)
    try:
        assert isinstance(engine, AsyncEngine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_create_async_engine_connects(settings: Settings) -> None:
    engine = create_async_engine(settings)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_async_sessionmaker_produces_async_session(settings: Settings) -> None:
    engine = create_async_engine(settings)
    try:
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as session:
            assert isinstance(session, AsyncSession)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_db_yields_session_and_closes(settings: Settings) -> None:
    engine = create_async_engine(settings)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    gen = get_db(maker)
    session = await gen.__anext__()
    assert isinstance(session, AsyncSession)

    # After the generator is exhausted, the session should be closed
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    await engine.dispose()


@pytest.mark.asyncio
async def test_init_db_creates_tables(settings: Settings) -> None:
    engine = create_async_engine(settings)
    try:
        await init_db(engine)

        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' ORDER BY table_name"
                )
            )
            tables = {row[0] for row in result}

        assert {"users", "daily_energy", "consumption_log", "pending_analyses"} <= tables
    finally:
        await engine.dispose()
