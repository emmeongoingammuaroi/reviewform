"""Node 4: Format the final review output.

Takes the classified issues and produces a clean, structured summary.
This is a pure data transformation — no LLM call, no external service.
"""

from app.core.logging import get_logger
from app.agent.state import ReviewState

logger = get_logger(__name__)


async def format_response(state: ReviewState) -> dict:
    """Format the final review summary with issue counts by severity."""
    issues = state.get("issues", [])
    existing_summary = state.get("summary", "")

    # Count issues by severity
    severity_counts: dict[str, int] = {}
    for issue in issues:
        sev = issue.get("severity", "suggestion")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Build a structured summary
    parts = [existing_summary]

    if severity_counts:
        counts_str = ", ".join(f"{count} {sev}" for sev, count in severity_counts.items())
        parts.append(f"\nIssue breakdown: {counts_str}")
    else:
        parts.append("\nNo issues found — code looks good!")

    summary = "\n".join(parts)

    logger.info("format_response.done", total_issues=len(issues), severities=severity_counts)

    return {"summary": summary}
