"""MCP Client — connects the LangGraph agent to the MCP server.

KEY CONCEPT — Why route through MCP instead of calling services directly?

The naive approach is: agent node → app.services.github (direct SDK call).
This works, but it makes the MCP server decorative — nothing actually uses it.

The correct architecture is:
    agent node → MCP client → (HTTP/SSE) → MCP server → services

This means:
1. The MCP server is LOAD-BEARING, not decoration
2. The same tools are available to Claude Desktop, other agents, or any MCP client
3. Agent nodes don't know or care HOW tools are implemented
4. You can swap tool implementations without touching agent code

This module provides `call_tool()` — the single function that agent nodes
use to invoke MCP tools. It handles connection lifecycle automatically.
"""

from mcp import ClientSession
from mcp.client.sse import sse_client

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def call_tool(tool_name: str, arguments: dict) -> str:
    """Call an MCP tool and return the text result.

    This connects to the MCP server via SSE, invokes the named tool,
    and returns the concatenated text content.

    Args:
        tool_name: Name of the MCP tool (e.g. "fetch_github_pr_diff")
        arguments: Tool arguments as a dict

    Returns:
        The text content returned by the tool.

    Raises:
        RuntimeError: If the tool returns an error or no content.
    """
    server_url = f"{settings.mcp_server_url}/sse"

    logger.info("mcp_client.calling", tool=tool_name, server=server_url)

    async with sse_client(server_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the MCP connection (handshake)
            await session.initialize()

            # Call the tool
            result = await session.call_tool(tool_name, arguments)

    if result.isError:
        error_text = result.content[0].text if result.content else "Unknown error"
        logger.error("mcp_client.tool_error", tool=tool_name, error=error_text)
        raise RuntimeError(f"MCP tool '{tool_name}' failed: {error_text}")

    # Concatenate all text content blocks
    texts = [block.text for block in result.content if hasattr(block, "text")]
    response_text = "\n".join(texts)

    logger.info(
        "mcp_client.success",
        tool=tool_name,
        response_length=len(response_text),
    )

    return response_text


async def fetch_github_pr_diff(pr_ref: str) -> dict:
    """Convenience wrapper: fetch a GitHub PR diff via MCP.

    Returns:
        Dict with key "diff" containing the PR diff text.
    """
    diff_text = await call_tool("fetch_github_pr_diff", {"pr_ref": pr_ref})
    return {"diff": diff_text}


async def search_coding_standards(query: str, limit: int = 5) -> list[str]:
    """Convenience wrapper: search coding standards via MCP.

    Returns:
        List of relevant coding standard texts.
    """
    result_text = await call_tool("search_coding_standards", {"query": query, "limit": limit})

    if result_text == "No matching standards found.":
        return []

    # The MCP tool joins standards with "---" separator
    return [s.strip() for s in result_text.split("---") if s.strip()]
