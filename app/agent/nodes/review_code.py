"""Node 3: AI-powered code review using gpt-4o.

This is the core node — it sends the code + retrieved standards to the LLM
and asks for a structured review.

KEY CONCEPT — Prompt Engineering for Agents:
The prompt is carefully structured to produce parseable output. We ask the LLM
to return JSON so we can extract individual issues. The retrieved coding standards
are injected as context, making this the "Augmented Generation" part of RAG.
"""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.state import ReviewState
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an expert code reviewer. \
Analyze the provided code and return a JSON object with your findings.

You MUST return valid JSON with this exact structure:
{
    "issues": [
        {
            "severity": "critical|warning|suggestion",
            "category": "security|performance|style|logic|best-practice",
            "line": <line_number_or_null>,
            "description": "<what's wrong>",
            "suggestion": "<how to fix it>"
        }
    ],
    "summary": "<2-3 sentence overall assessment>"
}

Guidelines:
- severity: "critical" = bugs/security holes, "warning" = should fix, "suggestion" = nice to have
- Be specific about line numbers when possible
- Provide actionable suggestions, not vague advice
- If the code looks good, return an empty issues list with a positive summary
"""


def _build_review_prompt(code: str, standards: list[str], language: str | None) -> str:
    """Build the review prompt with code and retrieved standards."""
    parts = []

    if language:
        parts.append(f"Language: {language}")

    if standards:
        parts.append("Relevant coding standards to check against:")
        for i, standard in enumerate(standards, 1):
            parts.append(f"  {i}. {standard}")

    parts.append(f"\nCode to review:\n```\n{code}\n```")
    parts.append("\nReturn your analysis as JSON.")

    return "\n".join(parts)


async def review_code(state: ReviewState) -> dict:
    """Send code + standards to gpt-4o and parse the structured response."""
    code = state["code"]
    standards = state.get("standards", [])
    language = state.get("language")

    logger.info("review_code.start", code_length=len(code), num_standards=len(standards))

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.1,  # Low temperature for consistent, factual reviews
    )

    user_prompt = _build_review_prompt(code, standards, language)

    start = time.monotonic()
    response = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    llm_latency_ms = (time.monotonic() - start) * 1000

    raw_review = str(response.content)

    # Parse the JSON response
    try:
        # Strip markdown code fences if present
        text = raw_review.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # remove first line
            text = text.rsplit("```", 1)[0]  # remove last fence

        parsed = json.loads(text)
        issues = parsed.get("issues", [])
        summary = parsed.get("summary", "Review completed.")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("review_code.parse_error", error=str(e))
        issues = []
        summary = raw_review  # Fall back to raw text

    logger.info("review_code.done", num_issues=len(issues))

    return {
        "raw_review": raw_review,
        "issues": issues,
        "summary": summary,
        "llm_latency_ms": llm_latency_ms,
    }
