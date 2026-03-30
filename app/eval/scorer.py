"""Eval scorer — LLM-as-judge + heuristic scoring for review quality.

This is what turns a log table into an evaluation framework. Two scoring methods:

1. **Heuristic scoring** (fast, deterministic, free):
   - Checks structural quality of the review (valid JSON? issues found? suggestions present?)
   - Catches obvious failures without burning LLM tokens

2. **LLM-as-judge scoring** (slower, nuanced, costs tokens):
   - Uses GPT-4o to evaluate whether the review is accurate, actionable, and complete
   - Scores on a rubric: relevance, accuracy, actionability, completeness
   - This is the industry-standard approach for evaluating LLM outputs

Together they give you a composite score that tracks model improvement over time.
"""

import json
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------

def _is_valid_json_review(raw_review: str) -> bool:
    """Check if the raw review is parseable JSON with expected structure."""
    try:
        text = raw_review.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        parsed = json.loads(text)
        return isinstance(parsed.get("issues"), list) and "summary" in parsed
    except (json.JSONDecodeError, KeyError, IndexError):
        return False


def score_heuristic(
    *,
    raw_review: str,
    issues: list[dict],
    code: str,
) -> dict:
    """Score a review using fast, deterministic heuristics.

    Returns:
        {
            "total": float (0.0-1.0),
            "breakdown": {
                "valid_json": float,
                "has_issues_or_clean_summary": float,
                "issues_have_suggestions": float,
                "issues_have_line_numbers": float,
                "severity_distribution": float,
            }
        }
    """
    breakdown = {}

    # 1. Valid JSON structure (0.2) — did the LLM follow instructions?
    breakdown["valid_json"] = 1.0 if _is_valid_json_review(raw_review) else 0.0

    # 2. Non-empty review (0.2) — did it find issues OR give a clean summary?
    if issues:
        breakdown["has_issues_or_clean_summary"] = 1.0
    elif raw_review and len(raw_review) > 20:
        breakdown["has_issues_or_clean_summary"] = 0.8  # clean code, still reviewed
    else:
        breakdown["has_issues_or_clean_summary"] = 0.0

    # 3. Issues have actionable suggestions (0.2)
    if issues:
        with_suggestion = sum(1 for i in issues if i.get("suggestion"))
        breakdown["issues_have_suggestions"] = with_suggestion / len(issues)
    else:
        breakdown["issues_have_suggestions"] = 1.0  # no issues = n/a, full marks

    # 4. Issues reference line numbers (0.2) — specificity check
    if issues:
        with_lines = sum(1 for i in issues if i.get("line") is not None)
        breakdown["issues_have_line_numbers"] = with_lines / len(issues)
    else:
        breakdown["issues_have_line_numbers"] = 1.0

    # 5. Severity distribution (0.2) — not everything is "critical" or "suggestion"
    if len(issues) >= 2:
        severities = {i.get("severity") for i in issues}
        breakdown["severity_distribution"] = min(len(severities) / 2.0, 1.0)
    else:
        breakdown["severity_distribution"] = 1.0  # too few issues to judge

    weights = {
        "valid_json": 0.2,
        "has_issues_or_clean_summary": 0.2,
        "issues_have_suggestions": 0.2,
        "issues_have_line_numbers": 0.2,
        "severity_distribution": 0.2,
    }

    total = sum(breakdown[k] * weights[k] for k in breakdown)

    return {"total": round(total, 3), "breakdown": breakdown}


# ---------------------------------------------------------------------------
# LLM-as-judge scoring
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI code reviews. Score the following code review on four dimensions.

For each dimension, assign a score from 0.0 to 1.0:

1. **relevance** — Are the issues relevant to the actual code? No hallucinated problems?
2. **accuracy** — Are the identified issues genuine bugs/problems? Are severity levels appropriate?
3. **actionability** — Are suggestions specific and implementable? Could a developer act on them immediately?
4. **completeness** — Did the review catch the important issues? Any obvious problems missed?

Return ONLY valid JSON:
{
    "relevance": <float>,
    "accuracy": <float>,
    "actionability": <float>,
    "completeness": <float>,
    "reasoning": "<1-2 sentence explanation>"
}
"""


async def score_llm_judge(
    *,
    code: str,
    raw_review: str,
    standards: list[str] | None = None,
) -> dict:
    """Score a review using GPT-4o as a judge.

    Returns:
        {
            "total": float (0.0-1.0),
            "breakdown": {
                "relevance": float,
                "accuracy": float,
                "actionability": float,
                "completeness": float,
            },
            "reasoning": str,
            "latency_ms": float,
        }
    """
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.0,  # deterministic judging
    )

    user_prompt_parts = [
        "## Code Under Review",
        f"```\n{code[:3000]}\n```",  # cap to avoid token bloat
        "",
        "## AI Review Output",
        raw_review[:3000],
    ]

    if standards:
        user_prompt_parts.extend([
            "",
            "## Coding Standards Used",
            "\n".join(f"- {s[:200]}" for s in standards[:5]),
        ])

    user_prompt_parts.append("\nScore this review on the four dimensions.")

    start = time.monotonic()
    response = await llm.ainvoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(user_prompt_parts)),
    ])
    latency_ms = (time.monotonic() - start) * 1000

    # Parse judge response
    try:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        parsed = json.loads(text)
        breakdown = {
            "relevance": float(parsed.get("relevance", 0)),
            "accuracy": float(parsed.get("accuracy", 0)),
            "actionability": float(parsed.get("actionability", 0)),
            "completeness": float(parsed.get("completeness", 0)),
        }
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("llm_judge.parse_error", error=str(e))
        breakdown = {
            "relevance": 0.0,
            "accuracy": 0.0,
            "actionability": 0.0,
            "completeness": 0.0,
        }
        reasoning = f"Judge parse error: {e}"

    # Equal weighting across dimensions
    total = sum(breakdown.values()) / 4.0

    return {
        "total": round(total, 3),
        "breakdown": breakdown,
        "reasoning": reasoning,
        "latency_ms": round(latency_ms, 1),
    }


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

async def score_review(
    *,
    code: str,
    raw_review: str,
    issues: list[dict],
    standards: list[str] | None = None,
    use_llm_judge: bool = True,
) -> dict:
    """Run both heuristic and (optionally) LLM-as-judge scoring.

    Returns:
        {
            "composite_score": float (0.0-1.0),
            "heuristic": { ... },
            "llm_judge": { ... } | None,
            "score_method": "heuristic+llm" | "heuristic",
        }
    """
    heuristic = score_heuristic(raw_review=raw_review, issues=issues, code=code)

    llm_judge = None
    if use_llm_judge:
        try:
            llm_judge = await score_llm_judge(
                code=code,
                raw_review=raw_review,
                standards=standards,
            )
        except Exception as e:
            logger.error("score_review.llm_judge_failed", error=str(e))

    # Composite: 40% heuristic + 60% LLM judge (if available)
    if llm_judge:
        composite = 0.4 * heuristic["total"] + 0.6 * llm_judge["total"]
        method = "heuristic+llm"
    else:
        composite = heuristic["total"]
        method = "heuristic"

    return {
        "composite_score": round(composite, 3),
        "heuristic": heuristic,
        "llm_judge": llm_judge,
        "score_method": method,
    }
