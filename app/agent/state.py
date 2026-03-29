"""LangGraph state definition for the code review workflow.

KEY CONCEPT — LangGraph State:
In LangGraph, the "state" is a TypedDict (or Pydantic model) that flows through
every node in your graph. Think of it like a request context that accumulates data
as it passes through each processing step.

Each node receives the full state, does its work, and returns a *partial* update.
LangGraph merges these updates automatically.

This is analogous to how FastAPI's dependency injection builds up request context,
but here the "context" is the evolving review analysis.
"""

from __future__ import annotations

from typing import TypedDict


class ReviewIssue(TypedDict):
    """A single issue found during code review."""

    severity: str  # critical / warning / suggestion
    category: str  # security / performance / style / logic / best-practice
    line: int | None
    description: str
    suggestion: str | None


class ReviewState(TypedDict, total=False):
    """The shared state that flows through the review graph.

    Fields are added progressively by each node:
    1. fetch_diff          -> sets code, pr_metadata
    2. retrieve_standards  -> sets standards
    3. review_code         -> sets issues, raw_review, summary
    4. format_response     -> updates summary with severity counts
    5. human_review        -> sets human_approved, human_comments
    6. log_eval            -> writes to PostgreSQL (no state change)
    """

    # --- Input (set at graph entry by the API route) ---
    session_id: str
    input_type: str  # "snippet" or "github_pr"
    raw_input: str  # code snippet or PR URL (validated by Pydantic schema)
    language: str | None

    # --- After fetch_diff ---
    code: str  # the actual code to review (PR diff or raw snippet)
    pr_metadata: dict | None  # PR URL, etc. if from GitHub

    # --- After retrieve_standards ---
    standards: list[str]  # relevant coding standard chunks from Qdrant (via MCP)

    # --- After review_code ---
    raw_review: str  # raw LLM output before structuring
    issues: list[ReviewIssue]
    summary: str
    llm_latency_ms: float | None  # LLM call duration (for eval logging)

    # --- After format_response ---
    # summary is updated with severity breakdown

    # --- After human_review (human-in-the-loop) ---
    human_approved: bool | None
    human_comments: str | None

    # --- Error handling ---
    error: str | None
