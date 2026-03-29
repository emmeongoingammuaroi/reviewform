"""Unit tests for the review graph nodes."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agent.nodes.format_response import format_response
from app.agent.nodes.fetch_diff import fetch_diff
from app.agent.nodes.retrieve_standards import retrieve_standards
from app.agent.nodes.human_review import human_review
from app.agent.nodes.log_eval import log_eval


# ---------------------------------------------------------------------------
# format_response tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_format_response_with_issues():
    """format_response node should count issues by severity."""
    state = {
        "issues": [
            {"severity": "critical", "category": "security", "description": "SQL injection"},
            {"severity": "warning", "category": "style", "description": "Long function"},
            {"severity": "critical", "category": "logic", "description": "Off-by-one"},
        ],
        "summary": "Found some issues.",
    }
    result = await format_response(state)
    assert "2 critical" in result["summary"]
    assert "1 warning" in result["summary"]


@pytest.mark.asyncio
async def test_format_response_no_issues():
    """format_response node should report clean code when no issues found."""
    state = {"issues": [], "summary": "Code looks clean."}
    result = await format_response(state)
    assert "No issues found" in result["summary"]


@pytest.mark.asyncio
async def test_format_response_preserves_existing_summary():
    """format_response should append to existing summary, not replace it."""
    state = {"issues": [], "summary": "AI assessment here."}
    result = await format_response(state)
    assert "AI assessment here." in result["summary"]
    assert "No issues found" in result["summary"]


# ---------------------------------------------------------------------------
# fetch_diff tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_diff_snippet():
    """fetch_diff should pass through raw code for snippet input."""
    state = {
        "input_type": "snippet",
        "raw_input": "def hello(): pass",
        "language": "python",
    }
    result = await fetch_diff(state)
    assert result["code"] == "def hello(): pass"
    assert result["pr_metadata"] is None


@pytest.mark.asyncio
async def test_fetch_diff_github_pr(mock_mcp_client):
    """fetch_diff should call MCP server for GitHub PR input."""
    mock_mcp_client.return_value = "diff --git a/file.py b/file.py\n+new line"

    state = {
        "input_type": "github_pr",
        "raw_input": "owner/repo/pull/42",
    }
    result = await fetch_diff(state)
    assert "diff --git" in result["code"]
    assert result["pr_metadata"]["url"] == "owner/repo/pull/42"
    mock_mcp_client.assert_called_once()


# ---------------------------------------------------------------------------
# retrieve_standards tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_standards_calls_mcp(mock_mcp_client):
    """retrieve_standards should call MCP search tool."""
    mock_mcp_client.return_value = "Standard 1\n---\nStandard 2"

    state = {"code": "def foo(): pass", "language": "python"}
    result = await retrieve_standards(state)
    assert len(result["standards"]) == 2
    assert "Standard 1" in result["standards"]


@pytest.mark.asyncio
async def test_retrieve_standards_empty_results(mock_mcp_client):
    """retrieve_standards should return empty list when no standards match."""
    mock_mcp_client.return_value = "No matching standards found."

    state = {"code": "x = 1", "language": None}
    result = await retrieve_standards(state)
    assert result["standards"] == []


@pytest.mark.asyncio
async def test_retrieve_standards_truncates_long_code(mock_mcp_client):
    """retrieve_standards should truncate code to 2000 chars for embedding."""
    mock_mcp_client.return_value = "No matching standards found."

    long_code = "x = 1\n" * 1000  # ~6000 chars
    state = {"code": long_code, "language": None}
    await retrieve_standards(state)

    # Check the query sent to MCP was truncated
    call_args = mock_mcp_client.call_args
    query_sent = call_args[1]["query"] if "query" in call_args[1] else call_args[0][1]["query"]
    assert len(query_sent) <= 2000


# ---------------------------------------------------------------------------
# human_review tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_review_returns_empty():
    """human_review should return empty dict (it's a checkpoint, not a processor)."""
    state = {
        "session_id": "abc-123",
        "issues": [{"severity": "warning", "description": "test"}],
    }
    result = await human_review(state)
    assert result == {}


# ---------------------------------------------------------------------------
# log_eval tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_eval_skips_without_raw_review():
    """log_eval should skip logging if no raw_review in state."""
    state = {"session_id": "abc-123", "raw_review": ""}
    result = await log_eval(state)
    assert result == {}


@pytest.mark.asyncio
async def test_log_eval_with_data():
    """log_eval should write to database when raw_review is present."""
    state = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "raw_review": '{"issues": [], "summary": "Looks good"}',
        "issues": [],
        "code": "x = 1",
        "standards": ["Be clean"],
        "language": "python",
        "llm_latency_ms": 1234.5,
    }

    # Mock the database session
    mock_db = AsyncMock()
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.agent.nodes.log_eval.async_session_factory", return_value=mock_session_ctx):
        result = await log_eval(state)

    assert result == {}
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()

    # Verify the EvalLog was created with latency
    eval_log = mock_db.add.call_args[0][0]
    assert eval_log.latency_ms == 1234.5
    assert eval_log.node_name == "review_code"
