"""Pydantic schemas for API request/response validation.

URL validation for GitHub PRs lives HERE, not in a graph node.
The graph should only receive clean, validated data.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class InputType(str, Enum):
    SNIPPET = "snippet"
    GITHUB_PR = "github_pr"


# Matches "owner/repo/pull/123" or full GitHub URL
_PR_REF_PATTERN = re.compile(
    r"^(?:https?://github\.com/)?[\w.-]+/[\w.-]+/pull/\d+/?$"
)


# --- Requests ---


class ReviewRequest(BaseModel):
    """What the user sends to start a code review.

    Validation rules:
    - If input_type is github_pr, content must be a valid PR reference
      (e.g. "owner/repo/pull/123" or full GitHub URL)
    - If input_type is snippet, content is the raw code (min 1 char)
    """

    input_type: InputType
    content: str = Field(
        ...,
        description="Code snippet or GitHub PR URL (e.g. 'owner/repo/pull/123')",
        min_length=1,
    )
    language: str | None = Field(None, description="Programming language hint (e.g. 'python')")

    @model_validator(mode="after")
    def validate_content_matches_type(self) -> ReviewRequest:
        """Ensure PR URLs are valid when input_type is github_pr."""
        if self.input_type == InputType.GITHUB_PR:
            if not _PR_REF_PATTERN.match(self.content.strip()):
                raise ValueError(
                    "For github_pr input, content must be a PR reference like "
                    "'owner/repo/pull/123' or 'https://github.com/owner/repo/pull/123'"
                )
        return self


class HumanFeedback(BaseModel):
    """Human-in-the-loop feedback on a review.

    Note: session_id comes from the URL path parameter, not the body.
    """

    approved: bool
    comments: str | None = None


# --- Responses ---


class ReviewIssue(BaseModel):
    """A single issue found during review."""

    severity: str  # critical / warning / suggestion
    category: str  # security / performance / style / logic / best-practice
    line: int | None = None
    description: str
    suggestion: str | None = None


class ReviewResponse(BaseModel):
    """The full review result returned to the user."""

    session_id: uuid.UUID
    status: str
    summary: str
    issues: list[ReviewIssue]
    standards_used: list[str]  # which coding standards were retrieved via RAG
    created_at: datetime


class EvalScoreRequest(BaseModel):
    """Request to score a specific eval log entry."""

    eval_id: uuid.UUID
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str | None = None
