"""MCP Server — exposes GitHub and Qdrant tools via Model Context Protocol.

KEY CONCEPT — MCP (Model Context Protocol):
MCP is a standard created by Anthropic for giving LLMs access to tools.
Instead of hardcoding tool calls in your agent, you expose tools as an
MCP server. Any MCP-compatible client can then discover and use them.

Why build an MCP server?
1. Reusability: The same tools work with Claude Desktop, your LangGraph agent,
   or any other MCP client
2. Separation of concerns: Tool implementation is decoupled from agent logic
3. Portfolio value: Shows you understand tool-use architecture

This server exposes two tools:
- fetch_github_pr_diff: Fetches a GitHub PR diff
- search_coding_standards: Queries Qdrant for relevant coding standards

Architecture note:
This runs as a SEPARATE PROCESS from FastAPI. It exposes an SSE endpoint
so the LangGraph agent (running inside FastAPI) can connect as an MCP client.
This is why it lives in its own top-level package (not inside app/).

Transport options:
- SSE (default): HTTP-based, used by the FastAPI agent and other HTTP clients
- stdio: Used by Claude Desktop and other stdio-based MCP clients
"""

import argparse
import asyncio

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from app.core.logging import setup_logging, get_logger
from mcp_server.tools import github, qdrant_search

logger = get_logger(__name__)

# Create the MCP server
server = Server("reviewflow-mcp")

# Registry of all tools — add new tools here
_TOOLS = {
    github.TOOL_DEFINITION["name"]: github,
    qdrant_search.TOOL_DEFINITION["name"]: qdrant_search,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the tools this server exposes."""
    return [Tool(**tool_mod.TOOL_DEFINITION) for tool_mod in _TOOLS.values()]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route tool invocations to the correct handler."""
    logger.info("mcp.tool_call", tool=name)

    tool_module = _TOOLS.get(name)
    if tool_module is None:
        raise ValueError(f"Unknown tool: {name}")

    return await tool_module.handle(arguments)


# ---------------------------------------------------------------------------
# SSE transport (HTTP-based) — used by the LangGraph agent
# ---------------------------------------------------------------------------

def create_sse_app() -> Starlette:
    """Create a Starlette app that serves the MCP server over SSE.

    KEY CONCEPT — SSE Transport:
    SSE (Server-Sent Events) lets the MCP server run as an HTTP service.
    The agent connects to /sse to establish a persistent connection,
    then sends tool calls to /messages. This is how the agent talks to
    the MCP server without stdio.

    Endpoints:
    - GET  /sse       — SSE stream (client connects here)
    - POST /messages  — tool call messages (client sends here)
    - GET  /health    — health check for Docker/load balancers
    """
    sse_transport = SseServerTransport("/messages")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )

    async def handle_health(request):
        from starlette.responses import JSONResponse
        return JSONResponse({"status": "healthy", "server": "reviewflow-mcp"})

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse_transport.handle_post_message),
            Route("/health", endpoint=handle_health),
        ],
    )


# ---------------------------------------------------------------------------
# stdio transport — used by Claude Desktop and other stdio MCP clients
# ---------------------------------------------------------------------------

async def run_stdio_server() -> None:
    """Run the MCP server over stdio (for Claude Desktop compatibility)."""
    logger.info("mcp.stdio_server_starting")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint — supports both SSE and stdio transports.

    Usage:
        python -m mcp_server.server                  # SSE on port 8001
        python -m mcp_server.server --port 9000      # SSE on custom port
        python -m mcp_server.server --transport stdio # stdio for Claude Desktop
    """
    parser = argparse.ArgumentParser(description="ReviewFlow MCP Server")
    parser.add_argument(
        "--transport", choices=["sse", "stdio"], default="sse",
        help="Transport protocol (default: sse)",
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Port for SSE transport (default: 8001)",
    )
    args = parser.parse_args()

    setup_logging()

    if args.transport == "stdio":
        asyncio.run(run_stdio_server())
    else:
        import uvicorn
        logger.info("mcp.sse_server_starting", port=args.port)
        app = create_sse_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
