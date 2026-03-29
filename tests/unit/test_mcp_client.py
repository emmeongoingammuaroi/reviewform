"""Unit tests for the MCP client wrapper."""

import pytest

from app.agent.mcp_client import (
    fetch_github_pr_diff,
    search_coding_standards,
)


@pytest.mark.asyncio
async def test_fetch_github_pr_diff_returns_dict(mock_mcp_client):
    """fetch_github_pr_diff should return dict with 'diff' key."""
    mock_mcp_client.return_value = "diff content here"
    result = await fetch_github_pr_diff("owner/repo/pull/1")
    assert result == {"diff": "diff content here"}


@pytest.mark.asyncio
async def test_search_coding_standards_parses_separator(mock_mcp_client):
    """search_coding_standards should split results on '---' separator."""
    mock_mcp_client.return_value = "Standard A\n---\nStandard B\n---\nStandard C"
    result = await search_coding_standards("some code")
    assert len(result) == 3
    assert result[0] == "Standard A"
    assert result[2] == "Standard C"


@pytest.mark.asyncio
async def test_search_coding_standards_empty(mock_mcp_client):
    """search_coding_standards should return empty list for no results."""
    mock_mcp_client.return_value = "No matching standards found."
    result = await search_coding_standards("obscure query")
    assert result == []


@pytest.mark.asyncio
async def test_search_coding_standards_single_result(mock_mcp_client):
    """search_coding_standards with one result (no separator)."""
    mock_mcp_client.return_value = "Only one standard"
    result = await search_coding_standards("query")
    assert result == ["Only one standard"]
