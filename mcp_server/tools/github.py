"""MCP Tool: Fetch GitHub PR diffs.

This tool is exposed via the MCP server so that any MCP-compatible client
(Claude Desktop, LangGraph agent, etc.) can fetch PR diffs for code review.
"""

from mcp.types import TextContent

from app.services.github import fetch_pr_diff


TOOL_DEFINITION = {
    "name": "fetch_github_pr_diff",
    "description": (
        "Fetch the diff of a GitHub Pull Request. "
        "Input: PR reference like 'owner/repo/pull/123'."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "pr_ref": {
                "type": "string",
                "description": "PR reference: 'owner/repo/pull/123' or full GitHub URL",
            },
        },
        "required": ["pr_ref"],
    },
}


async def handle(arguments: dict) -> list[TextContent]:
    """Execute the fetch_github_pr_diff tool."""
    pr_ref = arguments["pr_ref"]
    result = await fetch_pr_diff(pr_ref)
    return [TextContent(type="text", text=result["diff"])]
