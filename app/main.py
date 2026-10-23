"""FastAPI application entry point for DiabetaBot.

The ASGI app serves the ``/health`` endpoint that the docker compose
healthcheck probes. When an :class:`~app.runtime.AppRuntime` is supplied, the
FastAPI lifespan also starts the Telegram long-polling loop and the
midnight-WIB reset scheduler for the process lifetime, so a single
``uvicorn`` process serves both (PRD §8.1, D12).

The runtime is injected rather than built here so tests can start the app with
a fake. Production uses :func:`create_production_app` as a uvicorn factory
(see ``SPEC-deployment.md``).
"""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.routers.health import router as health_router

if TYPE_CHECKING:
    from app.runtime import AppRuntime


def parse_allowed_hosts(raw: str) -> list[str]:
    """Normalize a comma-separated hostname setting into a middleware list."""
    hosts = [part.strip().lower() for part in raw.split(",")]
    hosts = [host for host in hosts if host]
    if "testserver" not in hosts:
        hosts.append("testserver")
    return hosts


def create_app(
    *,
    runtime: "AppRuntime | None" = None,
    production: bool = False,
    allowed_hosts: Sequence[str] | None = None,
) -> FastAPI:
    """Build and configure the FastAPI application.

    Parameters
    ----------
    runtime:
        Optional process runtime. When given, its ``start``/``stop`` bracket
        the application lifespan; when ``None`` the app only serves HTTP.
    production:
        Disable interactive docs and enforce the ``allowed_hosts`` middleware.
    allowed_hosts:
        Hosts permitted to reach the app; required (non-empty) when
        ``production`` is set. The loopback ``testserver`` alias is always
        allowed so TestClient works.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            await runtime.start()
        try:
            yield
        finally:
            if runtime is not None:
                await runtime.stop()

    application = FastAPI(
        title="DiabetaBot",
        description="Telegram chatbot for personal nutrition and glycemic feasibility analysis.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        openapi_url=None if production else "/openapi.json",
    )
    if production:
        hosts = list(allowed_hosts or [])
        if "testserver" not in hosts:
            hosts.append("testserver")
        application.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    application.include_router(health_router)
    return application


def create_production_app() -> FastAPI:
    """uvicorn factory: build the app wired to the real services."""
    from app.config import Settings
    from app.runtime import build_runtime

    settings = Settings()
    hosts = parse_allowed_hosts(settings.allowed_hosts)
    return create_app(runtime=build_runtime(settings), production=True, allowed_hosts=hosts)


app = create_app()
