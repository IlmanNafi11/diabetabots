"""FastAPI application entry point for DiabetaBot.

The Telegram long-polling engine is started separately (see SPEC section 4);
this module exposes the ASGI app plus the /health endpoint that the
docker compose healthcheck probes.
"""

from fastapi import FastAPI

from app.routers.health import router as health_router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title="DiabetaBot",
        description="Telegram chatbot for personal nutrition and glycemic feasibility analysis.",
        version="0.1.0",
    )
    application.include_router(health_router)
    return application


app = create_app()
