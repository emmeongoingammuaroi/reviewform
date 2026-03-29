"""Unit tests for API routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Health endpoint should return 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_auth_token_generation(client):
    """POST /auth/token should return a valid JWT."""
    response = await client.post(
        "/api/v1/auth/token",
        json={"user_id": "test-user", "email": "test@example.com"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_auth_token_missing_user_id(client):
    """POST /auth/token without user_id should fail validation."""
    response = await client.post("/api/v1/auth/token", json={"email": "test@example.com"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_review_requires_auth(client):
    """POST /reviews/ without auth should return 403."""
    response = await client.post(
        "/api/v1/reviews/",
        json={"input_type": "snippet", "content": "print('hello')"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_review_invalid_pr_url(client, auth_headers):
    """POST /reviews/ with invalid PR URL should fail schema validation."""
    response = await client.post(
        "/api/v1/reviews/",
        headers=auth_headers,
        json={"input_type": "github_pr", "content": "not-a-valid-url"},
    )
    assert response.status_code == 422
    assert "PR reference" in response.json()["detail"][0]["msg"]


@pytest.mark.asyncio
async def test_review_valid_pr_url_format(client, auth_headers):
    """POST /reviews/ with valid PR format should pass schema validation.

    Note: The actual review will fail because the graph needs MCP + DB,
    but the schema validation should pass (no 422).
    """
    response = await client.post(
        "/api/v1/reviews/",
        headers=auth_headers,
        json={"input_type": "github_pr", "content": "owner/repo/pull/123"},
    )
    # Should NOT be a 422 validation error — it may be 500 from graph execution
    # but the input validation passed
    assert response.status_code != 422
