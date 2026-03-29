# ReviewFlow — AI-Powered Code Review Agent

A production-grade code review agent built with **LangGraph**, **FastAPI**, **RAG (Qdrant)**, and **MCP**. It analyzes code snippets or GitHub PR diffs using a multi-step AI workflow, retrieves relevant coding standards via vector search, provides structured feedback, and supports human-in-the-loop refinement.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Clients                                  │
│    (curl / Postman / Frontend / Claude Desktop / MCP clients)    │
└──────────────┬──────────────────────────────┬────────────────────┘
               │ HTTP/REST                    │ MCP (SSE or stdio)
               ▼                              ▼
┌──────────────────────────┐    ┌───────────────────────────────┐
│    FastAPI Backend         │    │    MCP Server (port 8001)      │
│   (JWT auth, async)        │    │   2 tools:                     │
│                            │    │   - fetch_github_pr_diff       │
│   POST /reviews/           │    │   - search_coding_standards    │
│   POST /reviews/{id}/      │    │                                │
│        feedback             │    │   Transports:                  │
│   GET  /eval/logs          │    │   - SSE (for LangGraph agent)  │
│   POST /eval/score         │    │   - stdio (for Claude Desktop) │
└──────────┬─────────────────┘    └──────────┬────────────────────┘
           │                                  │
           ▼                                  │
┌──────────────────────────────────────────┐  │
│       LangGraph Review Workflow           │  │
│                                           │  │
│  Agent nodes call tools via MCP client:   │  │
│                                           │  │
│  ┌────────────┐    ┌──────────────────┐  │  │
│  │ fetch_diff  │───▶│ retrieve_        │  │  │
│  │ (MCP call)  │    │ standards        │  │  │
│  └────────────┘    │ (MCP call → RAG) │  │  │
│                    └────────┬─────────┘  │  │
│                             ▼             │  │
│  ┌────────────┐    ┌──────────────────┐  │  │
│  │ log_eval   │◀───│ human_review     │  │  │
│  │ (Postgres) │    │ (checkpoint)     │  │  │
│  └─────┬──────┘    └────────▲────────┘  │  │
│        │           ┌────────┘            │  │
│        ▼           │                     │  │
│       END   ┌──────┴─────────┐           │  │
│             │ format_response │           │  │
│             └────────▲───────┘           │  │
│                      │                   │  │
│             ┌────────┴───────┐           │  │
│             │  review_code    │           │  │
│             │  (GPT-4o)      │           │  │
│             └────────────────┘           │  │
└──────────────────────┬───────────────────┘  │
                       │ MCP Client (SSE)      │
                       └───────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
       ┌──────────────┐  ┌──────────────┐  ┌────────────┐
       │  PostgreSQL   │  │   Qdrant     │  │  OpenAI    │
       │  - sessions   │  │   (vectors)  │  │  (gpt-4o)  │
       │  - eval_logs  │  │   coding     │  │  embeddings│
       └──────────────┘  │   standards  │  └────────────┘
                         └──────────────┘
```

### Key Architectural Decision: Nodes Call MCP, Not Services

The dependency flow is:

```
agent nodes  →  MCP client  →  (SSE/HTTP)  →  mcp_server/  →  services/
```

NOT:

```
agent nodes  →  services/   ← WRONG — bypasses MCP entirely
```

`services/github.py` and `services/qdrant.py` are low-level SDK wrappers. They live in `app/` because the MCP server imports them — but **agent nodes never touch them directly**. If they did, the MCP server would be decoration and the architecture story falls apart.

This means the MCP server is **load-bearing**: it's the only path for the agent to access external data. The same tools work with Claude Desktop, other AI agents, or any MCP client.

## Project Structure

```
reviewflow/
├── app/                          # FastAPI application
│   ├── main.py                   # App entrypoint, lifespan, route registration
│   ├── core/                     # Cross-cutting infrastructure
│   │   ├── config.py             # Pydantic Settings (env-based config)
│   │   ├── dependencies.py       # JWT auth + DB session injection
│   │   ├── logging.py            # Structured logging (structlog)
│   │   └── middleware.py         # Request ID tracking
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth.py           # POST /auth/token
│   │   │   ├── review.py         # POST /reviews, POST /reviews/{id}/feedback
│   │   │   └── eval.py           # GET /eval/logs, POST /eval/score, GET /eval/summary
│   │   └── schemas/
│   │       └── review.py         # Pydantic models + input validation
│   ├── agent/                    # LangGraph workflow (the AI brain)
│   │   ├── state.py              # ReviewState — shared TypedDict flowing through nodes
│   │   ├── builder.py            # Graph construction, edges, checkpointing
│   │   ├── mcp_client.py         # MCP client — how nodes call tools via MCP server
│   │   └── nodes/                # Each node = one step in the review pipeline
│   │       ├── fetch_diff.py     # Node 1: Fetch PR diff via MCP tool
│   │       ├── retrieve_standards.py  # Node 2: RAG via MCP tool
│   │       ├── review_code.py    # Node 3: GPT-4o structured review
│   │       ├── format_response.py # Node 4: Format summary with severity counts
│   │       ├── human_review.py   # Node 5: Pause for human feedback
│   │       └── log_eval.py       # Node 6: Persist eval data to PostgreSQL
│   ├── db/
│   │   ├── base.py               # Async SQLAlchemy engine + session factory
│   │   └── models.py             # ReviewSession + EvalLog ORM models
│   └── services/                 # Low-level SDK wrappers (used by MCP server ONLY)
│       ├── github.py             # GitHub API client (fetch PR diffs)
│       └── qdrant.py             # Qdrant vector search client
│
├── mcp_server/                   # MCP server (separate process + container)
│   ├── server.py                 # Server entrypoint, SSE + stdio transports
│   └── tools/
│       ├── github.py             # Tool: fetch_github_pr_diff
│       └── qdrant_search.py      # Tool: search_coding_standards
│
├── alembic/                      # Database migrations
│   ├── env.py                    # Async migration runner
│   ├── script.py.mako            # Migration template
│   └── versions/                 # Migration scripts
│       └── 001_initial_schema.py
├── standards_data/
│   └── standards.json            # Sample coding standards for RAG
├── scripts/
│   └── seed_standards.py         # Seed Qdrant with coding standards
├── tests/
│   ├── conftest.py               # Shared fixtures (auth, client, MCP mock)
│   ├── unit/                     # Fast, isolated tests
│   └── integration/              # Tests hitting real services
├── docker/
│   ├── Dockerfile.api            # FastAPI container (dev + prod stages)
│   ├── Dockerfile.mcp            # MCP server container (SSE on port 8001)
│   └── nginx.conf                # Reverse proxy config
├── docker-compose.yml            # API + MCP + PostgreSQL + Qdrant
├── docker-compose.override.yml   # Local dev overrides
├── alembic.ini                   # Database migrations config
├── pyproject.toml                # Dependencies & tool config
├── .env.example                  # Environment variable template
└── .github/workflows/ci.yml     # Lint + test + type-check
```

### Why this structure?

**`app/` vs `mcp_server/`** — The MCP server runs as a **separate process** with its own container. The FastAPI agent connects to it via SSE as an MCP client. Keeping it in its own top-level package reflects deployment reality: two containers, two entrypoints, one shared `services/` layer.

**`app/agent/mcp_client.py`** — The bridge between agent nodes and the MCP server. Nodes call `mcp_client.fetch_github_pr_diff()` or `mcp_client.search_coding_standards()`, which connect to the MCP server via SSE. This ensures nodes never import `services/` directly.

**`app/services/`** — Low-level SDK wrappers for GitHub API and Qdrant. These are imported **only by `mcp_server/tools/`**, never by agent nodes. Think of them as the "implementation detail" behind MCP tools.

**`app/core/`** — Groups infrastructure that every module depends on: config, auth, logging, middleware.

**`app/api/schemas/`** — Pydantic models with validation logic (e.g., GitHub PR URL format). Validation lives here, not in graph nodes — the graph only receives clean data.

**`docker/`** — Per-service Dockerfiles. The API and MCP server have different dependencies and startup commands.

## How the LangGraph Workflow Works

The review pipeline is a **6-node directed graph** built with LangGraph:

### State

Every node reads from and writes to a shared `ReviewState` (a TypedDict):

```python
class ReviewState(TypedDict, total=False):
    session_id: str          # Set at entry
    code: str                # Set by fetch_diff (via MCP)
    standards: list[str]     # Set by retrieve_standards (via MCP -> RAG)
    issues: list[dict]       # Set by review_code (GPT-4o)
    summary: str             # Set by review_code, refined by format_response
    human_approved: bool     # Set by human feedback
```

### Nodes

| Node | What it does | Calls |
|------|-------------|-------|
| `fetch_diff` | Fetches PR diff via MCP, or passes through raw snippet | MCP: `fetch_github_pr_diff` |
| `retrieve_standards` | Queries Qdrant for relevant coding standards via MCP | MCP: `search_coding_standards` |
| `review_code` | Sends code + standards to GPT-4o, parses structured JSON | OpenAI API |
| `format_response` | Formats summary with issue severity breakdown | Pure logic (no I/O) |
| `human_review` | Checkpoint — graph pauses for human approval | LangGraph interrupt |
| `log_eval` | Persists prompt/response/scores to PostgreSQL | Database write |

### Graph Flow

```
fetch_diff -> retrieve_standards -> review_code -> format_response -> human_review
                                        ^                                |
                                        |                                |
                                        +--- (if rejected) -------------+
                                                                         |
                                                      (if approved) -> log_eval -> END
```

### Human-in-the-Loop

LangGraph supports **checkpointing**: the graph pauses at `human_review` (via `interrupt_before`) and saves its state. When the user submits feedback via `POST /reviews/{id}/feedback`, the graph resumes from exactly where it left off. If rejected, it loops back to `review_code` with the human's comments as additional context.

## MCP Server

The MCP server exposes two tools and supports two transport protocols:

| Tool | Description |
|------|-------------|
| `fetch_github_pr_diff` | Fetches a GitHub PR diff given `owner/repo/pull/123` |
| `search_coding_standards` | Queries Qdrant for coding standards relevant to a code snippet |

| Transport | Used by | Command |
|-----------|---------|---------|
| **SSE** (default) | LangGraph agent, HTTP clients | `python -m mcp_server.server` |
| **stdio** | Claude Desktop, stdio MCP clients | `python -m mcp_server.server --transport stdio` |

## Evaluation Framework

The `log_eval` graph node writes every LLM interaction to PostgreSQL:
- **Prompt** sent to the model
- **Response** received
- **Latency** in milliseconds
- **Score** (0.0-1.0, assigned later via API)
- **Node name** (which graph step produced it)

This lets you answer: *"Is my agent getting better or worse over time?"*

**API endpoints:**
- `GET /eval/logs` — Browse logged interactions
- `POST /eval/score` — Score a specific log entry
- `GET /eval/summary` — Aggregate metrics (avg score, avg latency per node)

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose

### 1. Clone and configure

```bash
git clone https://github.com/your-username/reviewflow.git
cd reviewflow
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start everything with Docker

```bash
docker compose up
```

This starts: API (port 8000), MCP server (port 8001), PostgreSQL (5432), Qdrant (6333).

### 3. Or run locally

```bash
# Start infrastructure
docker compose up -d postgres qdrant

# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Seed the vector store
python -m scripts.seed_standards

# Start MCP server (terminal 1)
python -m mcp_server.server

# Start API (terminal 2)
uvicorn app.main:app --reload
```

### 4. Try it out

```bash
# Get a token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "dev"}' | jq -r '.access_token')

# Submit a code review
curl -X POST http://localhost:8000/api/v1/reviews/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "snippet",
    "content": "def process(data):\n    result = eval(data)\n    return result",
    "language": "python"
  }'
```

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API Framework | FastAPI | Async-native, auto-generated docs, Pydantic integration |
| Agent Framework | LangGraph | State machines with checkpointing, human-in-the-loop |
| LLM | OpenAI GPT-4o | Best structured output quality for code review |
| Vector Store | Qdrant | Fast, async-native, purpose-built for embeddings |
| Database | PostgreSQL + SQLAlchemy | Async ORM, robust for eval logging |
| Tool Protocol | MCP (SSE + stdio) | Standard for exposing tools to AI clients |
| Logging | structlog | Structured JSON logs for production observability |
| Auth | JWT (python-jose) | Stateless, simple for portfolio scope |
| CI/CD | GitHub Actions | Lint (ruff) + test (pytest) + type-check (mypy) |
| Containers | Docker Compose | One command to spin up the full stack |

## License

MIT
