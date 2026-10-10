"""Health endpoint used by the docker compose healthcheck."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Report process liveness for container orchestration."""
    return {"status": "ok"}
