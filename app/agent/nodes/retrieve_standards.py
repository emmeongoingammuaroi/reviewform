"""Node 2: Retrieve relevant coding standards via RAG.

This node queries the MCP server for coding standards relevant to the
code being reviewed. The MCP server handles the actual Qdrant vector search.

KEY CONCEPT — RAG in an Agent Workflow:
Instead of stuffing all coding standards into the prompt (which wastes tokens
and may exceed context limits), we use vector search to find only the
relevant ones. The LLM then reviews the code *with those standards in context*.

The dependency flow is:
    this node → mcp_client → (SSE) → mcp_server → services/qdrant.py → Qdrant
"""

from app.core.logging import get_logger
from app.agent.state import ReviewState
from app.agent import mcp_client

logger = get_logger(__name__)


async def retrieve_standards(state: ReviewState) -> dict:
    """Query coding standards via MCP server's Qdrant search tool.

    Uses the code snippet as the search query to find semantically similar
    coding standards, best practices, and style guidelines.
    """
    code = state["code"]
    language = state.get("language")

    logger.info("retrieve_standards.start", language=language)

    # Build a search query from the code + language hint
    query = code[:2000]  # Truncate for embedding — first 2000 chars is usually enough
    if language:
        query = f"[{language}] {query}"

    # Call MCP server instead of Qdrant directly
    standards = await mcp_client.search_coding_standards(query=query, limit=5)

    logger.info("retrieve_standards.done", num_standards=len(standards))

    return {"standards": standards}
