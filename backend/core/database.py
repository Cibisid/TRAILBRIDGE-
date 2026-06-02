"""
TrialBridge — Database Layer
Async SQLAlchemy setup with PostgreSQL + pgvector extension.

Why async? Our API handles multiple concurrent requests.
Async DB means the server doesn't sit idle waiting for PostgreSQL —
it handles other requests while waiting for DB responses.
This is how production APIs are built.
"""

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# -----------------------------------------------
# Engine
# -----------------------------------------------
# pool_size: how many connections to keep open (production: 20)
# max_overflow: extra connections allowed under load
# pool_pre_ping: test connection before using it (prevents stale connection errors)
# echo: log every SQL statement (only in development)

engine = create_async_engine(
    settings.database_url,
    pool_size=5 if settings.is_development else 20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=settings.is_development,
    # NullPool in testing so we don't leak connections
    poolclass=NullPool if settings.app_env == "testing" else None,
)


# -----------------------------------------------
# Session Factory
# -----------------------------------------------
# autocommit=False: we control when to commit (safer — prevents partial writes)
# autoflush=False: we control when to flush (better performance)
# expire_on_commit=False: keep objects accessible after commit

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# -----------------------------------------------
# Base Model
# -----------------------------------------------
class Base(DeclarativeBase):
    """
    All SQLAlchemy ORM models inherit from this.
    Provides type-aware column mapping and metadata.
    """
    pass


# -----------------------------------------------
# pgvector setup
# -----------------------------------------------
async def enable_pgvector(connection: Any) -> None:
    """
    Enable the pgvector PostgreSQL extension.
    This gives us vector similarity search directly in the DB —
    no separate Pinecone or Weaviate service needed for V1.
    """
    await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    logger.info("pgvector extension enabled")


# -----------------------------------------------
# Database Initialization
# -----------------------------------------------
async def init_db() -> None:
    """
    Called once at application startup.
    Creates all tables and enables pgvector.
    In production, we use Alembic migrations instead —
    but we run this here as a safety net.
    """
    async with engine.begin() as conn:
        # Enable pgvector before creating tables
        await enable_pgvector(conn)
        # Import all models so Base knows about them
        from backend.models import audit, match, patient, trial  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")


async def close_db() -> None:
    """Called at application shutdown to clean up connections."""
    await engine.dispose()
    logger.info("Database connections closed")


# -----------------------------------------------
# Dependency — used in FastAPI route handlers
# -----------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.
    The 'async with' ensures the session is always closed,
    even if an exception occurs mid-request.

    Usage in a route:
        @router.get("/trials")
        async def get_trials(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
