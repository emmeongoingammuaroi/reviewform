"""Shared test fixtures for the ReviewFlow test suite.

KEY CONCEPT — Test Fixtures:
Fixtures provide reusable setup/teardown for tests. Instead of each test
creating its own DB session, auth token, and HTTP client, they share these
fixtures. This keeps tests DRY and makes them easier to write.

Usage in tests:
    async def test_something(client, auth_headers):
        response = await client.post("/api/v1/reviews/", headers=auth_headers, ...)
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient
from jose import jwt

# Load test env vars BEFORE importing app modules so Settings() picks them up.
_env_test = Path(__file__).resolve().parent.parent / ".env.test"
load_dotenv(_env_test, override=True)

from app.core.config import settings  # noqa: E402
from app.db.base import async_session_factory, get_db  # noqa: E402
from app.main import app  # noqa: E402

# ---------------------------------------------------------------------------
# Auth fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_user() -> dict:
    """A test user payload."""
    return {"user_id": "test-user-1", "email": "test@example.com"}


@pytest.fixture
def auth_token(test_user: dict) -> str:
    """Generate a valid JWT token for testing."""
    payload = {
        "sub": test_user["user_id"],
        "email": test_user["email"],
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


@pytest.fixture
def auth_headers(auth_token: str) -> dict:
    """HTTP headers with a valid Bearer token."""
    return {"Authorization": f"Bearer {auth_token}"}


# ---------------------------------------------------------------------------
# Database fixtures — isolated per test via savepoint rollback
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_session():
    """Database session that rolls back after each test.

    Uses nested transactions (savepoints) so each test gets a clean slate
    without needing to drop/recreate tables.
    """
    async with async_session_factory() as session:
        async with session.begin():
            yield session
            await session.rollback()


@pytest.fixture
def _override_db(db_session):  # type: ignore[no-untyped-def]
    """Override FastAPI's get_db dependency with the test session."""

    async def _get_test_db():  # type: ignore[no-untyped-def]
        yield db_session

    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# HTTP client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# MCP client mock — prevents tests from needing a live MCP server
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_mcp_client():
    """Mock the MCP client so tests don't need a running MCP server.

    This is autouse=True because almost every test would fail without it.
    Tests that need real MCP calls should override this fixture.
    """
    with patch("app.agent.mcp_client.call_tool", new_callable=AsyncMock) as mock:
        # Default: return empty results
        mock.return_value = ""
        yield mock
