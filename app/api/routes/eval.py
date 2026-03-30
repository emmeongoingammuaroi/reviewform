"""Eval routes — view, score, and report on evaluation logs.

Now integrates with the eval framework:
- scorer.py: automated scoring (heuristic + LLM-as-judge)
- report.py: aggregate metrics, trends, regression detection
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import EvalLog
from app.api.schemas.review import EvalScoreRequest
from app.eval import scorer, report

router = APIRouter(prefix="/eval", tags=["eval"])


# ---------------------------------------------------------------------------
# Log browsing
# ---------------------------------------------------------------------------


@router.get("/logs")
async def list_eval_logs(
    session_id: uuid.UUID | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List eval logs, optionally filtered by session."""
    stmt = select(EvalLog).order_by(EvalLog.created_at.desc()).limit(limit)
    if session_id:
        stmt = stmt.where(EvalLog.session_id == session_id)

    result = await db.execute(stmt)
    logs = result.scalars().all()

    return [
        {
            "id": str(log.id),
            "session_id": str(log.session_id),
            "node_name": log.node_name,
            "prompt": log.prompt[:200] + "..." if len(log.prompt) > 200 else log.prompt,
            "response": log.response[:200] + "..." if len(log.response) > 200 else log.response,
            "score": log.score,
            "score_method": log.score_method,
            "score_reason": log.score_reason,
            "heuristic_scores": log.heuristic_scores,
            "llm_judge_scores": log.llm_judge_scores,
            "latency_ms": log.latency_ms,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


# ---------------------------------------------------------------------------
# Manual scoring (existing)
# ---------------------------------------------------------------------------


@router.post("/score")
async def score_eval_log(
    request: EvalScoreRequest,
    db: AsyncSession = Depends(get_db),
):
    """Score an eval log entry (manual evaluation)."""
    result = await db.execute(select(EvalLog).where(EvalLog.id == request.eval_id))
    log = result.scalar_one_or_none()

    if not log:
        raise HTTPException(status_code=404, detail="Eval log not found")

    log.score = request.score
    log.score_reason = request.reason
    log.score_method = "manual"
    await db.commit()

    return {"id": str(log.id), "score": log.score, "score_method": "manual", "score_reason": log.score_reason}


# ---------------------------------------------------------------------------
# Automated scoring (NEW)
# ---------------------------------------------------------------------------


@router.post("/auto-score/{eval_id}")
async def auto_score_eval_log(
    eval_id: uuid.UUID,
    use_llm_judge: bool = Query(default=True, description="Include LLM-as-judge (costs tokens)"),
    db: AsyncSession = Depends(get_db),
):
    """Run automated scoring (heuristic + optional LLM-as-judge) on an eval log.

    This is the key differentiator: automated scoring that can run at scale
    without human intervention, enabling tracking of model improvement over time.
    """
    result = await db.execute(select(EvalLog).where(EvalLog.id == eval_id))
    log = result.scalar_one_or_none()

    if not log:
        raise HTTPException(status_code=404, detail="Eval log not found")

    # Parse issues from the stored review JSON for accurate heuristic scoring
    issues = []
    try:
        text = log.response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        parsed = json.loads(text)
        issues = parsed.get("issues", [])
    except (json.JSONDecodeError, KeyError, IndexError):
        pass  # heuristic scorer handles empty issues gracefully

    scores = await scorer.score_review(
        code=log.prompt,
        raw_review=log.response,
        issues=issues,
        use_llm_judge=use_llm_judge,
    )

    log.score = scores["composite_score"]
    log.score_method = scores["score_method"]
    log.score_reason = scores.get("llm_judge", {}).get("reasoning") if scores.get("llm_judge") else None
    log.heuristic_scores = scores["heuristic"]
    log.llm_judge_scores = scores.get("llm_judge")
    await db.commit()

    return {
        "id": str(log.id),
        "composite_score": scores["composite_score"],
        "score_method": scores["score_method"],
        "heuristic": scores["heuristic"],
        "llm_judge": scores.get("llm_judge"),
    }


# ---------------------------------------------------------------------------
# Reporting (NEW)
# ---------------------------------------------------------------------------


@router.get("/summary")
async def eval_summary(db: AsyncSession = Depends(get_db)):
    """Get aggregate eval metrics — average score by node."""
    return await report.summary_by_node(db)


@router.get("/trend")
async def eval_trend(
    days: int = Query(default=30, ge=1, le=365),
    bucket: str = Query(default="day", pattern="^(day|week)$"),
    db: AsyncSession = Depends(get_db),
):
    """Score trend over time — track model improvement or regression."""
    return await report.score_trend(db, days=days, bucket=bucket)


@router.get("/distribution")
async def eval_distribution(db: AsyncSession = Depends(get_db)):
    """Score distribution across quality tiers (excellent/good/fair/poor)."""
    return await report.score_distribution(db)


@router.get("/regression")
async def eval_regression(
    window: int = Query(default=10, ge=2, le=100),
    threshold: float = Query(default=0.1, ge=0.01, le=0.5),
    db: AsyncSession = Depends(get_db),
):
    """Check if recent review scores have regressed compared to historical baseline."""
    return await report.regression_check(db, window=window, threshold=threshold)


@router.get("/report")
async def eval_full_report(db: AsyncSession = Depends(get_db)):
    """Full evaluation report — summary, trends, distribution, and regression check."""
    return await report.full_report(db)
