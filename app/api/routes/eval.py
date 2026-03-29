"""Eval routes — view and score evaluation logs."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import EvalLog
from app.api.schemas.review import EvalScoreRequest

router = APIRouter(prefix="/eval", tags=["eval"])


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
            "score_reason": log.score_reason,
            "latency_ms": log.latency_ms,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


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
    await db.commit()

    return {"id": str(log.id), "score": log.score, "score_reason": log.score_reason}


@router.get("/summary")
async def eval_summary(db: AsyncSession = Depends(get_db)):
    """Get aggregate eval metrics — average score by node, over time."""
    stmt = (
        select(
            EvalLog.node_name,
            func.count(EvalLog.id).label("count"),
            func.avg(EvalLog.score).label("avg_score"),
            func.avg(EvalLog.latency_ms).label("avg_latency_ms"),
        )
        .where(EvalLog.score.isnot(None))
        .group_by(EvalLog.node_name)
    )

    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "node_name": row.node_name,
            "count": row.count,
            "avg_score": round(float(row.avg_score), 3) if row.avg_score else None,
            "avg_latency_ms": round(float(row.avg_latency_ms), 1) if row.avg_latency_ms else None,
        }
        for row in rows
    ]
