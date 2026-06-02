"""
TrialBridge — Health Check Endpoint

Every production service needs a /health endpoint.
Load balancers, Kubernetes, and monitoring tools ping this
to know if the service is alive and ready to handle traffic.

We expose two levels:
- /health       — liveness: is the process running?
- /health/ready — readiness: can it handle requests? (DB connected, etc.)
"""

import time

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.core.database import get_db
from backend.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

# Track when the application started
_start_time = time.time()


@router.get("/health", summary="Liveness check")
async def health_check():
    """
    Liveness probe — confirms the process is running.
    Returns 200 immediately without checking dependencies.
    Kubernetes uses this to know if it should restart the pod.
    """
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        "uptime_seconds": int(time.time() - _start_time),
    }


@router.get("/health/ready", summary="Readiness check")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Readiness probe — confirms the service can handle traffic.
    Checks all critical dependencies: database, (future: Redis, models).
    Returns 503 if any dependency is down — load balancer will
    stop sending traffic to this instance.
    """
    checks = {}

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        checks["database"] = f"error: {str(e)}"

    # Overall status
    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "uptime_seconds": int(time.time() - _start_time),
    }
