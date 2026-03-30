"""Node 6: Log the review session and LLM outputs for evaluation.

KEY CONCEPT — Evaluation as a First-Class Graph Node:
Instead of silently logging eval data as a side-effect inside review_code,
we make it an explicit step in the graph. This means:
1. Eval logging is visible in the graph diagram — no hidden side-effects
2. It runs AFTER the review is complete, so it captures the full picture
3. It can be skipped or swapped without touching review logic
4. It writes to PostgreSQL — the eval API endpoints read from the same table

This node delegates to app.eval.logger — the eval framework owns persistence.
Scores are assigned automatically via scorer.py or manually via POST /eval/score.
"""

from app.core.logging import get_logger
from app.eval.logger import log_review_eval
from app.agent.state import ReviewState

logger = get_logger(__name__)


async def log_eval(state: ReviewState) -> dict:
    """Persist the review results to the eval_logs table.

    Delegates to the eval framework's logger so the same persistence
    logic can be reused outside the graph (batch eval, CI pipelines).
    """
    eval_log = await log_review_eval(
        session_id=state.get("session_id", ""),
        code=state.get("code", ""),
        standards=state.get("standards", []),
        language=state.get("language"),
        raw_review=state.get("raw_review", ""),
        issues=state.get("issues", []),
        llm_latency_ms=state.get("llm_latency_ms"),
    )

    if eval_log:
        logger.info("log_eval.done", eval_log_id=str(eval_log.id))

    return {}
