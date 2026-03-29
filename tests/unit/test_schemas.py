"""Unit tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from app.api.schemas.review import ReviewRequest, InputType, HumanFeedback, EvalScoreRequest


# ---------------------------------------------------------------------------
# ReviewRequest validation
# ---------------------------------------------------------------------------


class TestReviewRequest:
    def test_valid_snippet(self):
        """Snippet input with raw code should pass."""
        req = ReviewRequest(input_type="snippet", content="def hello(): pass")
        assert req.input_type == InputType.SNIPPET
        assert req.language is None

    def test_valid_snippet_with_language(self):
        """Snippet with language hint should pass."""
        req = ReviewRequest(input_type="snippet", content="x = 1", language="python")
        assert req.language == "python"

    def test_valid_pr_short_format(self):
        """PR reference in short format should pass."""
        req = ReviewRequest(input_type="github_pr", content="owner/repo/pull/123")
        assert req.input_type == InputType.GITHUB_PR

    def test_valid_pr_full_url(self):
        """PR reference as full GitHub URL should pass."""
        req = ReviewRequest(
            input_type="github_pr",
            content="https://github.com/owner/repo/pull/456",
        )
        assert req.content == "https://github.com/owner/repo/pull/456"

    def test_invalid_pr_url_rejects(self):
        """Invalid PR URL should fail validation."""
        with pytest.raises(ValidationError, match="PR reference"):
            ReviewRequest(input_type="github_pr", content="not-a-url")

    def test_invalid_pr_missing_pull(self):
        """PR URL without /pull/ should fail."""
        with pytest.raises(ValidationError, match="PR reference"):
            ReviewRequest(input_type="github_pr", content="owner/repo/123")

    def test_empty_content_rejects(self):
        """Empty content should fail min_length validation."""
        with pytest.raises(ValidationError):
            ReviewRequest(input_type="snippet", content="")

    def test_snippet_does_not_validate_url(self):
        """Snippet input should accept any non-empty string, even if it looks like a URL."""
        req = ReviewRequest(input_type="snippet", content="https://example.com")
        assert req.content == "https://example.com"


# ---------------------------------------------------------------------------
# EvalScoreRequest validation
# ---------------------------------------------------------------------------


class TestEvalScoreRequest:
    def test_valid_score(self):
        req = EvalScoreRequest(
            eval_id="550e8400-e29b-41d4-a716-446655440000", score=0.85
        )
        assert req.score == 0.85

    def test_score_too_high(self):
        with pytest.raises(ValidationError):
            EvalScoreRequest(
                eval_id="550e8400-e29b-41d4-a716-446655440000", score=1.5
            )

    def test_score_too_low(self):
        with pytest.raises(ValidationError):
            EvalScoreRequest(
                eval_id="550e8400-e29b-41d4-a716-446655440000", score=-0.1
            )
