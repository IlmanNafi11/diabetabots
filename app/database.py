"""Async database engine, session factory, and request-scoped session provider.

Business logic obtains sessions through :func:`get_db` (FastAPI dependency
injection, SPEC section 5.6). The engine is created from
:class:`app.config.Settings` so no credentials are hardcoded.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine as _create_async_engine

from app.config import Settings
from app.models.models import Base


def create_async_engine(settings: Settings) -> AsyncEngine:
    """Build an async engine from the configured ``database_url``.

    ``pool_pre_ping`` guards against connections dropped by an external
    PostgreSQL (Mode B) that may idle between polls.
    """
    return _create_async_engine(settings.database_url, pool_pre_ping=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to ``engine``."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a session and guarantee it is closed when the request ends."""
    async with sessionmaker() as session:
        yield session


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables declared on :class:`Base` if they do not yet exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
