"""Seed script — loads coding standards into Qdrant for RAG retrieval.

Run this once to populate the vector store with coding standards.
Usage: python -m scripts.seed_standards
"""

import asyncio
import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings


STANDARDS_DIR = Path(__file__).parent.parent / "standards_data"


async def seed():
    """Load coding standards from JSON files and upsert into Qdrant."""
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

    # Create collection if it doesn't exist
    collections = await client.get_collections()
    collection_names = [c.name for c in collections.collections]

    if settings.qdrant_collection not in collection_names:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"Created collection: {settings.qdrant_collection}")

    # Load standards from JSON file
    standards_file = STANDARDS_DIR / "standards.json"
    if not standards_file.exists():
        print(f"No standards file found at {standards_file}")
        print("Creating a sample standards file...")
        _create_sample_standards(standards_file)

    with open(standards_file) as f:
        standards = json.load(f)

    # Generate embeddings and upsert
    texts = [s["text"] for s in standards]
    vectors = await embeddings.aembed_documents(texts)

    points = [
        PointStruct(
            id=i,
            vector=vector,
            payload={"text": standard["text"], "category": standard.get("category", "general")},
        )
        for i, (standard, vector) in enumerate(zip(standards, vectors))
    ]

    await client.upsert(collection_name=settings.qdrant_collection, points=points)
    print(f"Seeded {len(points)} coding standards into Qdrant")


def _create_sample_standards(path: Path):
    """Create a sample coding standards file for demonstration."""
    sample_standards = [
        {
            "text": "Functions should have a single responsibility. If a function does more than one thing, split it into smaller functions.",
            "category": "clean-code",
        },
        {
            "text": "Always validate user input at API boundaries. Never trust data from external sources without sanitization.",
            "category": "security",
        },
        {
            "text": "Use parameterized queries for database operations. Never concatenate user input into SQL strings.",
            "category": "security",
        },
        {
            "text": "Handle exceptions explicitly. Avoid bare except clauses. Log the error with context before re-raising or returning an error response.",
            "category": "error-handling",
        },
        {
            "text": "Use type hints for function signatures in Python. This improves readability and enables static analysis tools.",
            "category": "python",
        },
        {
            "text": "Async functions that perform I/O should use async libraries (httpx, asyncpg) instead of blocking calls (requests, psycopg2).",
            "category": "python-async",
        },
        {
            "text": "API endpoints should return consistent response shapes. Use Pydantic models to enforce response schemas.",
            "category": "api-design",
        },
        {
            "text": "Avoid deeply nested code. Use early returns to reduce indentation and improve readability.",
            "category": "clean-code",
        },
        {
            "text": "Database queries in loops (N+1 problem) should be replaced with batch queries or JOINs.",
            "category": "performance",
        },
        {
            "text": "Sensitive data (API keys, passwords, tokens) must never be hardcoded. Use environment variables or secret management services.",
            "category": "security",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sample_standards, f, indent=2)
    print(f"Created sample standards at {path}")


if __name__ == "__main__":
    asyncio.run(seed())
