"""Node 5: Human-in-the-loop review checkpoint.

KEY CONCEPT — Human-in-the-Loop with LangGraph:
LangGraph supports "interrupts" — you can pause the graph at any node and
wait for external input. This is perfect for code review because:
1. The AI produces a review
2. The graph pauses and returns the review to the user
3. The user approves, rejects, or adds comments
4. The graph resumes with the human feedback

In our implementation, we handle this via the API:
- POST /review       -> starts the graph, it pauses at human_review node
- POST /review/{id}/feedback -> resumes the graph with human input

For now, this node simply marks the state as awaiting human input.
The actual interrupt/resume is handled by the graph builder.
"""

from app.agent.state import ReviewState
from app.core.logging import get_logger

logger = get_logger(__name__)


async def human_review(state: ReviewState) -> dict:
    """Checkpoint node for human review.

    This node is reached after the AI review is complete.
    The graph will be interrupted here (via LangGraph's interrupt mechanism)
    to wait for human feedback before continuing.
    """
    logger.info(
        "human_review.waiting",
        session_id=state.get("session_id"),
        num_issues=len(state.get("issues", [])),
    )

    # The actual human feedback will be injected when the graph resumes.
    # For now, return the current state as-is.
    return {}
