"""Unit tests for the eval scorer — heuristic and LLM-as-judge scoring."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.eval.scorer import score_heuristic, score_llm_judge, score_review

# ---------------------------------------------------------------------------
# Heuristic scorer tests
# ---------------------------------------------------------------------------


VALID_REVIEW = json.dumps(
    {
        "issues": [
            {
                "severity": "critical",
                "category": "security",
                "line": 5,
                "description": "Using eval() on user input",
                "suggestion": "Use ast.literal_eval() instead",
            },
            {
                "severity": "warning",
                "category": "style",
                "line": 10,
                "description": "Function too long",
                "suggestion": "Extract helper functions",
            },
        ],
        "summary": "Security issue found with eval usage.",
    }
)

CODE = "def process(data):\n    return eval(data)"


def test_heuristic_perfect_review():
    """A well-structured review with valid JSON, suggestions, and line numbers should score high."""
    issues = json.loads(VALID_REVIEW)["issues"]
    result = score_heuristic(raw_review=VALID_REVIEW, issues=issues, code=CODE)

    assert result["total"] >= 0.8
    assert result["breakdown"]["valid_json"] == 1.0
    assert result["breakdown"]["has_issues_or_clean_summary"] == 1.0
    assert result["breakdown"]["issues_have_suggestions"] == 1.0
    assert result["breakdown"]["issues_have_line_numbers"] == 1.0
    assert result["breakdown"]["severity_distribution"] == 1.0


def test_heuristic_invalid_json():
    """Invalid JSON review should lose the valid_json score."""
    result = score_heuristic(raw_review="Not valid JSON at all", issues=[], code=CODE)

    assert result["breakdown"]["valid_json"] == 0.0


def test_heuristic_no_suggestions():
    """Issues without suggestions should lose the suggestions score."""
    issues = [
        {"severity": "critical", "category": "security", "line": 5, "description": "Bad"},
        {"severity": "warning", "category": "style", "line": None, "description": "Also bad"},
    ]
    raw = json.dumps({"issues": issues, "summary": "Problems found."})

    result = score_heuristic(raw_review=raw, issues=issues, code=CODE)

    assert result["breakdown"]["issues_have_suggestions"] == 0.0


def test_heuristic_no_line_numbers():
    """Issues without line numbers should lose the line numbers score."""
    issues = [
        {
            "severity": "critical",
            "category": "security",
            "description": "Bad",
            "suggestion": "Fix it",
        },
    ]
    raw = json.dumps({"issues": issues, "summary": "Problems found."})

    result = score_heuristic(raw_review=raw, issues=issues, code=CODE)

    assert result["breakdown"]["issues_have_line_numbers"] == 0.0


def test_heuristic_empty_review():
    """Empty review should score poorly."""
    result = score_heuristic(raw_review="", issues=[], code=CODE)

    assert result["breakdown"]["valid_json"] == 0.0
    assert result["breakdown"]["has_issues_or_clean_summary"] == 0.0


def test_heuristic_clean_code_review():
    """A clean code review (no issues, good summary) should still score well."""
    raw = json.dumps({"issues": [], "summary": "Code looks clean and follows best practices."})

    result = score_heuristic(raw_review=raw, issues=[], code=CODE)

    assert result["total"] >= 0.8  # should not penalize clean reviews


def test_heuristic_single_severity():
    """All issues with same severity should get partial distribution score."""
    issues = [
        {
            "severity": "critical",
            "category": "security",
            "line": 1,
            "description": "A",
            "suggestion": "Fix A",
        },
        {
            "severity": "critical",
            "category": "logic",
            "line": 2,
            "description": "B",
            "suggestion": "Fix B",
        },
    ]
    raw = json.dumps({"issues": issues, "summary": "Multiple critical issues."})

    result = score_heuristic(raw_review=raw, issues=issues, code=CODE)

    assert result["breakdown"]["severity_distribution"] == 0.5


def test_heuristic_markdown_fenced_json():
    """Should handle JSON wrapped in markdown code fences."""
    fenced = f"```json\n{VALID_REVIEW}\n```"
    issues = json.loads(VALID_REVIEW)["issues"]

    result = score_heuristic(raw_review=fenced, issues=issues, code=CODE)

    assert result["breakdown"]["valid_json"] == 1.0


def test_heuristic_total_is_weighted_average():
    """Total score should be the weighted average of all breakdown scores."""
    issues = json.loads(VALID_REVIEW)["issues"]
    result = score_heuristic(raw_review=VALID_REVIEW, issues=issues, code=CODE)

    expected = sum(v * 0.2 for v in result["breakdown"].values())
    assert abs(result["total"] - round(expected, 3)) < 0.001


# ---------------------------------------------------------------------------
# LLM-as-judge tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_judge_parses_response():
    """LLM judge should parse a valid judge response and compute scores."""
    judge_response = json.dumps(
        {
            "relevance": 0.9,
            "accuracy": 0.8,
            "actionability": 0.85,
            "completeness": 0.7,
            "reasoning": "Good review with minor gaps.",
        }
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content=judge_response)

    with patch("app.eval.scorer.ChatOpenAI", return_value=mock_llm):
        result = await score_llm_judge(code=CODE, raw_review=VALID_REVIEW)

    assert result["breakdown"]["relevance"] == 0.9
    assert result["breakdown"]["accuracy"] == 0.8
    assert result["total"] == round((0.9 + 0.8 + 0.85 + 0.7) / 4, 3)
    assert result["reasoning"] == "Good review with minor gaps."
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_llm_judge_handles_parse_error():
    """LLM judge should return zeros if the judge response is unparseable."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="This is not JSON")

    with patch("app.eval.scorer.ChatOpenAI", return_value=mock_llm):
        result = await score_llm_judge(code=CODE, raw_review=VALID_REVIEW)

    assert result["total"] == 0.0
    assert "parse error" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_llm_judge_with_standards():
    """LLM judge should include standards in the prompt when provided."""
    judge_response = json.dumps(
        {
            "relevance": 0.9,
            "accuracy": 0.9,
            "actionability": 0.9,
            "completeness": 0.9,
            "reasoning": "Great.",
        }
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content=judge_response)

    with patch("app.eval.scorer.ChatOpenAI", return_value=mock_llm):
        await score_llm_judge(
            code=CODE,
            raw_review=VALID_REVIEW,
            standards=["Always validate input", "Never use eval()"],
        )

    # Verify standards were passed in the prompt
    call_args = mock_llm.ainvoke.call_args[0][0]
    user_msg = call_args[1].content
    assert "Coding Standards" in user_msg
    assert "Never use eval()" in user_msg


# ---------------------------------------------------------------------------
# Composite scorer tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_review_composite():
    """Composite score should blend heuristic (40%) and LLM judge (60%)."""
    judge_response = json.dumps(
        {
            "relevance": 1.0,
            "accuracy": 1.0,
            "actionability": 1.0,
            "completeness": 1.0,
            "reasoning": "Perfect.",
        }
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content=judge_response)

    issues = json.loads(VALID_REVIEW)["issues"]

    with patch("app.eval.scorer.ChatOpenAI", return_value=mock_llm):
        result = await score_review(
            code=CODE,
            raw_review=VALID_REVIEW,
            issues=issues,
            use_llm_judge=True,
        )

    assert result["score_method"] == "heuristic+llm"
    assert result["heuristic"] is not None
    assert result["llm_judge"] is not None
    # Composite = 0.4 * heuristic + 0.6 * llm
    expected = 0.4 * result["heuristic"]["total"] + 0.6 * result["llm_judge"]["total"]
    assert abs(result["composite_score"] - round(expected, 3)) < 0.001


@pytest.mark.asyncio
async def test_score_review_heuristic_only():
    """When use_llm_judge=False, should use heuristic only."""
    issues = json.loads(VALID_REVIEW)["issues"]

    result = await score_review(
        code=CODE,
        raw_review=VALID_REVIEW,
        issues=issues,
        use_llm_judge=False,
    )

    assert result["score_method"] == "heuristic"
    assert result["llm_judge"] is None
    assert result["composite_score"] == result["heuristic"]["total"]


@pytest.mark.asyncio
async def test_score_review_llm_failure_fallback():
    """If LLM judge fails, should fall back to heuristic-only scoring."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = Exception("API error")

    issues = json.loads(VALID_REVIEW)["issues"]

    with patch("app.eval.scorer.ChatOpenAI", return_value=mock_llm):
        result = await score_review(
            code=CODE,
            raw_review=VALID_REVIEW,
            issues=issues,
            use_llm_judge=True,
        )

    assert result["score_method"] == "heuristic"
    assert result["llm_judge"] is None
