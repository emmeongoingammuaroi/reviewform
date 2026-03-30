"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import auth, eval, review
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestIDMiddleware
from app.db.base import Base, engine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown events."""
    setup_logging()
    logger.info("app.starting", app_name=settings.app_name)

    # Create tables (in production, use Alembic migrations instead)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    await engine.dispose()
    logger.info("app.shutdown")


app = FastAPI(
    title=settings.app_name,
    description="AI-Powered Code Review Agent using LangGraph + RAG + MCP",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestIDMiddleware)

# Routes
app.include_router(auth.router, prefix="/api/v1")
app.include_router(review.router, prefix="/api/v1")
app.include_router(eval.router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "app": settings.app_name}
