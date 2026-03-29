"""Node 6: Log the review session and LLM outputs for evaluation.

KEY CONCEPT — Evaluation as a First-Class Graph Node:
Instead of silently logging eval data as a side-effect inside review_code,
we make it an explicit step in the graph. This means:
1. Eval logging is visible in the graph diagram — no hidden side-effects
2. It runs AFTER the review is complete, so it captures the full picture
3. It can be skipped or swapped without touching review logic
4. It writes to PostgreSQL — the eval API endpoints read from the same table

This node captures: prompt, response, latency, and the node that produced it.
Scores are assigned later via the POST /eval/score endpoint (human evaluation).
"""

import uuid

from app.core.logging import get_logger
from app.db.base import async_session_factory
from app.db.models import EvalLog
from app.agent.state import ReviewState

logger = get_logger(__name__)


async def log_eval(state: ReviewState) -> dict:
    """Persist the review results to the eval_logs table.

    Captures the raw LLM output from review_code so it can be scored later.
    """
    session_id = state.get("session_id", "")
    raw_review = state.get("raw_review", "")
    issues = state.get("issues", [])
    code = state.get("code", "")
    standards = state.get("standards", [])

    if not raw_review:
        logger.warning("log_eval.skip", reason="no raw_review in state")
        return {}

    logger.info("log_eval.start", session_id=session_id)

    # Reconstruct what was sent to the LLM (for eval traceability)
    prompt_summary = (
        f"Code length: {len(code)} chars\n"
        f"Standards used: {len(standards)}\n"
        f"Language: {state.get('language', 'unknown')}"
    )

    async with async_session_factory() as db:
        eval_log = EvalLog(
            id=uuid.uuid4(),
            session_id=uuid.UUID(session_id) if session_id else uuid.uuid4(),
            node_name="review_code",
            prompt=prompt_summary,
            response=raw_review,
            latency_ms=state.get("llm_latency_ms"),
        )
        db.add(eval_log)
        await db.commit()

    logger.info(
        "log_eval.done",
        session_id=session_id,
        num_issues=len(issues),
    )

    return {}
