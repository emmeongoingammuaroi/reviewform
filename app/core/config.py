"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central config — all values come from env vars or .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # App
    app_name: str = "ReviewFlow"
    debug: bool = False

    # Auth
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://reviewflow:reviewflow@localhost:5432/reviewflow"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "coding_standards"

    # GitHub
    github_token: str = ""

    # MCP Server (the agent connects to this as a client)
    mcp_server_url: str = "http://localhost:8001"


settings = Settings()
