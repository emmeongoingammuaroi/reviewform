"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central config — all values come from env vars or .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # App
    app_name: str = "ReviewFlow"
    debug: bool = False

    # Auth
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"

    # PostgreSQL
    database_url: str
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "coding_standards"

    # GitHub
    github_token: str

    # MCP Server (the agent connects to this as a client)
    mcp_server_url: str = "http://localhost:8001"

    # CORS
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # Request limits
    max_content_size: int = 100_000  # ~100KB max code snippet
    review_timeout_seconds: int = 300  # 5 min max for LLM review


settings = Settings()
