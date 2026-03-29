"""MCP Tool: Search coding standards via Qdrant vector search.

This tool is exposed via the MCP server so that any MCP-compatible client
can query the coding standards knowledge base using semantic search.
"""

from mcp.types import TextContent

from app.services.qdrant import search_standards


TOOL_DEFINITION = {
    "name": "search_coding_standards",
    "description": (
        "Search the coding standards knowledge base for best practices "
        "relevant to a code snippet or topic. Uses vector similarity search."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Code snippet or topic to search standards for",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of results (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


async def handle(arguments: dict) -> list[TextContent]:
    """Execute the search_coding_standards tool."""
    query = arguments["query"]
    limit = arguments.get("limit", 5)
    standards = await search_standards(query=query, limit=limit)
    text = "\n---\n".join(standards) if standards else "No matching standards found."
    return [TextContent(type="text", text=text)]
