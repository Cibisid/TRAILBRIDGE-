"""
TrialBridge — FastAPI Application Entry Point

This is where the entire application is assembled:
- Middleware (CORS, rate limiting, request logging)
- Routers (all API endpoints)
- Startup/shutdown lifecycle events
- Exception handlers
- Health check
- Prometheus metrics
"""

import time
import uuid
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from backend.api.v1.endpoints import auth, health, match, patient, trials
from backend.core.config import get_settings
from backend.core.database import close_db, init_db
from backend.core.logging import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)


# -----------------------------------------------
# Startup & Shutdown
# -----------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown.
    The code before 'yield' runs on startup.
    The code after 'yield' runs on shutdown.
    This is the modern FastAPI pattern (replaces @app.on_event).
    """
    # --- STARTUP ---
    configure_logging()
    logger.info("TrialBridge starting up", env=settings.app_env, version=settings.app_version)

    # Initialize Sentry error tracking (production only)
    if settings.sentry_dsn and settings.is_production:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.app_env,
            release=f"trialbridge@{settings.app_version}",
            traces_sample_rate=0.1,  # Sample 10% of transactions for performance
        )
        logger.info("Sentry error tracking initialized")

    # Initialize database (creates tables, enables pgvector)
    await init_db()
    logger.info("Database initialized")

    logger.info("TrialBridge startup complete — ready to serve requests")

    yield  # <-- Application runs here

    # --- SHUTDOWN ---
    logger.info("TrialBridge shutting down")
    await close_db()
    logger.info("Shutdown complete")


# -----------------------------------------------
# App Instance
# -----------------------------------------------
app = FastAPI(
    title="TrialBridge API",
    description="""
    AI-powered clinical trial matching platform.
    Connects patients to eligible trials through intelligent EHR parsing,
    semantic search, and explainable matching.
    """,
    version=settings.app_version,
    docs_url="/docs" if not settings.is_production else None,  # Hide docs in production
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
)


# -----------------------------------------------
# Middleware
# -----------------------------------------------

# 1. CORS — controls which origins can call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


# 2. Request ID + Timing middleware
# Attaches a unique ID to every request for tracing across logs
@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    # Add request context to all log entries within this request
    import structlog
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    logger.info("Request started")

    try:
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "Request completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"
        return response

    except Exception as exc:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error("Request failed", error=str(exc), duration_ms=duration_ms)
        raise


# -----------------------------------------------
# Exception Handlers
# -----------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url.path)},
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc) -> JSONResponse:
    logger.error("Unhandled server error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. This has been logged."},
    )


# -----------------------------------------------
# Prometheus Metrics
# -----------------------------------------------
# Automatically instruments all endpoints with:
# - Request count, latency, in-progress requests
# - Exposed at GET /metrics for Prometheus to scrape
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app)


# -----------------------------------------------
# Routers
# -----------------------------------------------
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix=settings.api_prefix, tags=["Authentication"])
app.include_router(patient.router, prefix=settings.api_prefix, tags=["Patient"])
app.include_router(trials.router, prefix=settings.api_prefix, tags=["Trials"])
app.include_router(match.router, prefix=settings.api_prefix, tags=["Matching"])
