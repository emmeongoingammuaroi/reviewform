"""Node 1: Fetch the code to review.

If the input is a GitHub PR URL, fetch the diff via the MCP server.
If it's a raw code snippet, pass it through directly.

KEY CONCEPT — Nodes call MCP, not services:
This node does NOT import app.services.github directly. Instead it calls
the MCP server's fetch_github_pr_diff tool via the MCP client. This keeps
the MCP server load-bearing — it's the real integration point, not decoration.

The dependency flow is:
    this node → mcp_client → (SSE) → mcp_server → services/github.py
"""

from app.core.logging import get_logger
from app.agent.state import ReviewState
from app.agent import mcp_client

logger = get_logger(__name__)


async def fetch_diff(state: ReviewState) -> dict:
    """Fetch code from GitHub PR (via MCP) or pass through raw snippet.

    Returns a partial state update with 'code' and 'pr_metadata'.
    """
    input_type = state["input_type"]
    raw_input = state["raw_input"]

    logger.info("fetch_diff.start", input_type=input_type)

    if input_type == "github_pr":
        # Call the MCP server to fetch the PR diff
        result = await mcp_client.fetch_github_pr_diff(raw_input)
        return {
            "code": result["diff"],
            "pr_metadata": {"url": raw_input},
        }
    else:
        # Raw code snippet — use as-is
        return {
            "code": raw_input,
            "pr_metadata": None,
        }
