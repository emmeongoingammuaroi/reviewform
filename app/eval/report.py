"""Eval report — aggregate metrics queries for tracking model improvement.

This module answers the questions an eval framework needs to answer:
- Is review quality improving or regressing over time?
- Which scoring dimensions are weakest?
- How does latency trend as we change models/prompts?
- Are there score regressions in recent reviews?

All queries are async and return dicts ready for API serialization.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import EvalLog

logger = get_logger(__name__)


async def summary_by_node(db: AsyncSession) -> list[dict[str, Any]]:
    """Average score and latency grouped by node — the original /eval/summary."""
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
    return [
        {
            "node_name": row.node_name,
            "count": row.count,
            "avg_score": round(float(row.avg_score), 3) if row.avg_score else None,
            "avg_latency_ms": round(float(row.avg_latency_ms), 1) if row.avg_latency_ms else None,
        }
        for row in result.all()
    ]


async def score_trend(
    db: AsyncSession,
    *,
    days: int = 30,
    bucket: str = "day",
) -> list[dict[str, Any]]:
    """Score trend over time — daily or weekly buckets.

    Returns time series: [{date, avg_score, avg_latency_ms, count}, ...]
    Tracks whether model quality is improving or regressing.
    """
    since = datetime.now(UTC) - timedelta(days=days)

    # Bucket by day or week
    if bucket == "week":
        date_trunc = func.date_trunc("week", EvalLog.created_at)
    else:
        date_trunc = func.date_trunc("day", EvalLog.created_at)

    stmt = (
        select(
            date_trunc.label("period"),
            func.count(EvalLog.id).label("count"),
            func.avg(EvalLog.score).label("avg_score"),
            func.avg(EvalLog.latency_ms).label("avg_latency_ms"),
        )
        .where(EvalLog.created_at >= since, EvalLog.score.isnot(None))
        .group_by("period")
        .order_by("period")
    )

    result = await db.execute(stmt)
    return [
        {
            "period": row.period.isoformat() if row.period else None,
            "count": row.count,
            "avg_score": round(float(row.avg_score), 3) if row.avg_score else None,
            "avg_latency_ms": round(float(row.avg_latency_ms), 1) if row.avg_latency_ms else None,
        }
        for row in result.all()
    ]


async def score_distribution(db: AsyncSession) -> dict[str, Any]:
    """Score distribution in buckets — how many reviews fall in each quality tier.

    Returns: {
        "excellent": count (0.8-1.0),
        "good": count (0.6-0.8),
        "fair": count (0.4-0.6),
        "poor": count (0.0-0.4),
        "unscored": count,
        "total": count,
    }
    """
    scored_stmt = select(
        func.count(case((EvalLog.score >= 0.8, EvalLog.id))).label("excellent"),
        func.count(case(((EvalLog.score >= 0.6) & (EvalLog.score < 0.8), EvalLog.id))).label(
            "good"
        ),
        func.count(case(((EvalLog.score >= 0.4) & (EvalLog.score < 0.6), EvalLog.id))).label(
            "fair"
        ),
        func.count(case(((EvalLog.score < 0.4), EvalLog.id))).label("poor"),
    ).where(EvalLog.score.isnot(None))

    unscored_stmt = select(func.count(EvalLog.id)).where(EvalLog.score.is_(None))
    total_stmt = select(func.count(EvalLog.id))

    scored_result = await db.execute(scored_stmt)
    row = scored_result.one()

    unscored_result = await db.execute(unscored_stmt)
    unscored = unscored_result.scalar_one()

    total_result = await db.execute(total_stmt)
    total = total_result.scalar_one()

    return {
        "excellent": row.excellent,
        "good": row.good,
        "fair": row.fair,
        "poor": row.poor,
        "unscored": unscored,
        "total": total,
    }


async def regression_check(
    db: AsyncSession,
    *,
    window: int = 10,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare recent scores against historical baseline to detect regressions.

    Compares the last `window` scored reviews against all prior reviews.
    Flags a regression if avg score dropped by more than `threshold`.

    Returns: {
        "regression_detected": bool,
        "recent_avg": float,
        "baseline_avg": float,
        "delta": float,
        "recent_count": int,
        "baseline_count": int,
    }
    """
    # Get recent scored logs
    recent_stmt = (
        select(EvalLog.score)
        .where(EvalLog.score.isnot(None))
        .order_by(EvalLog.created_at.desc())
        .limit(window)
    )
    recent_result = await db.execute(recent_stmt)
    recent_scores = [row[0] for row in recent_result.all()]

    if len(recent_scores) < 2:
        return {
            "regression_detected": False,
            "recent_avg": None,
            "baseline_avg": None,
            "delta": None,
            "recent_count": len(recent_scores),
            "baseline_count": 0,
        }

    recent_avg = sum(recent_scores) / len(recent_scores)

    # Baseline: everything except the recent window
    baseline_stmt = select(func.avg(EvalLog.score), func.count(EvalLog.id)).where(
        EvalLog.score.isnot(None)
    )
    baseline_result = await db.execute(baseline_stmt)
    baseline_row = baseline_result.one()
    baseline_avg = float(baseline_row[0]) if baseline_row[0] else 0.0
    baseline_count = baseline_row[1]

    # Subtract recent contribution from baseline for a clean comparison
    if baseline_count > len(recent_scores):
        adjusted_total = baseline_avg * baseline_count - sum(recent_scores)
        adjusted_count = baseline_count - len(recent_scores)
        baseline_avg = adjusted_total / adjusted_count
    else:
        # Not enough historical data for comparison
        baseline_avg = recent_avg

    delta = recent_avg - baseline_avg

    return {
        "regression_detected": delta < -threshold,
        "recent_avg": round(recent_avg, 3),
        "baseline_avg": round(baseline_avg, 3),
        "delta": round(delta, 3),
        "recent_count": len(recent_scores),
        "baseline_count": max(0, baseline_count - len(recent_scores)),
    }


async def full_report(db: AsyncSession) -> dict[str, Any]:
    """Generate a complete evaluation report combining all metrics.

    This is the single endpoint that answers: "How is our review quality?"
    """
    node_summary = await summary_by_node(db)
    trend = await score_trend(db, days=30, bucket="day")
    distribution = await score_distribution(db)
    regression = await regression_check(db)

    return {
        "by_node": node_summary,
        "trend_30d": trend,
        "distribution": distribution,
        "regression": regression,
    }
