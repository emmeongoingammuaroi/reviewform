"""Eval logger — persists review results for later scoring and reporting.

Extracted from the log_eval graph node so that:
1. The eval framework owns its own persistence logic
2. The graph node stays thin (delegates to this module)
3. Other callers (batch eval, CI pipelines) can log without the graph
"""

from __future__ import annotations

import uuid
from typing import Any

from app.core.logging import get_logger
from app.db.base import async_session_factory
from app.db.models import EvalLog

logger = get_logger(__name__)


async def log_review_eval(
    *,
    session_id: str,
    code: str,
    standards: list[str],
    language: str | None,
    raw_review: str,
    issues: list[dict[str, Any]],
    llm_latency_ms: float | None,
) -> EvalLog | None:
    """Persist a review result to the eval_logs table.

    Returns the created EvalLog, or None if skipped.
    """
    if not raw_review:
        logger.warning("eval_logger.skip", reason="no raw_review")
        return None

    logger.info("eval_logger.start", session_id=session_id)

    prompt_summary = (
        f"Code length: {len(code)} chars\n"
        f"Standards used: {len(standards)}\n"
        f"Language: {language or 'unknown'}"
    )

    async with async_session_factory() as db:
        eval_log = EvalLog(
            id=uuid.uuid4(),
            session_id=uuid.UUID(session_id) if session_id.strip() else uuid.uuid4(),
            node_name="review_code",
            prompt=prompt_summary,
            response=raw_review,
            latency_ms=llm_latency_ms,
        )
        db.add(eval_log)
        await db.commit()
        await db.refresh(eval_log)

    logger.info(
        "eval_logger.done",
        session_id=session_id,
        eval_log_id=str(eval_log.id),
        num_issues=len(issues),
    )

    return eval_log
