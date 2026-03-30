"""Qdrant service — vector search for coding standards (RAG retrieval)."""

from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Lazy-initialized clients
_qdrant_client: AsyncQdrantClient | None = None
_embeddings: OpenAIEmbeddings | None = None


def _get_qdrant() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _qdrant_client


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)  # type: ignore[arg-type]
    return _embeddings


async def search_standards(query: str, limit: int = 5) -> list[str]:
    """Search Qdrant for coding standards relevant to the query.

    Args:
        query: The code snippet or search query to find relevant standards.
        limit: Maximum number of results to return.

    Returns:
        List of coding standard text chunks, ranked by relevance.
    """
    client = _get_qdrant()
    embeddings = _get_embeddings()

    # Generate embedding for the query
    query_vector = await embeddings.aembed_query(query)

    # Search Qdrant
    results = await client.search(  # type: ignore[attr-defined]
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=limit,
        score_threshold=0.7,  # Only return reasonably relevant results
    )

    standards = [hit.payload.get("text", "") for hit in results if hit.payload]

    logger.info("qdrant.search_done", query_length=len(query), num_results=len(standards))

    return standards
