"""Unit tests for the eval report module — aggregate metrics and trend queries."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.eval.report import (
    summary_by_node,
    score_trend,
    score_distribution,
    regression_check,
    full_report,
)


def _make_mock_db(execute_results):
    """Create a mock AsyncSession that returns predefined query results."""
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=execute_results)
    return db


# ---------------------------------------------------------------------------
# summary_by_node tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_by_node():
    """Should return avg score and latency grouped by node."""
    mock_row = MagicMock()
    mock_row.node_name = "review_code"
    mock_row.count = 10
    mock_row.avg_score = 0.756
    mock_row.avg_latency_ms = 1234.5678

    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)

    result = await summary_by_node(db)

    assert len(result) == 1
    assert result[0]["node_name"] == "review_code"
    assert result[0]["count"] == 10
    assert result[0]["avg_score"] == 0.756
    assert result[0]["avg_latency_ms"] == 1234.6


@pytest.mark.asyncio
async def test_summary_by_node_empty():
    """Should return empty list when no scored logs exist."""
    mock_result = MagicMock()
    mock_result.all.return_value = []

    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)

    result = await summary_by_node(db)
    assert result == []


# ---------------------------------------------------------------------------
# score_trend tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_trend():
    """Should return time-series data bucketed by day."""
    now = datetime.now(timezone.utc)

    mock_row = MagicMock()
    mock_row.period = now
    mock_row.count = 5
    mock_row.avg_score = 0.8
    mock_row.avg_latency_ms = 500.0

    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)

    result = await score_trend(db, days=7, bucket="day")

    assert len(result) == 1
    assert result[0]["count"] == 5
    assert result[0]["avg_score"] == 0.8


# ---------------------------------------------------------------------------
# score_distribution tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_distribution():
    """Should return counts for each quality tier."""
    scored_row = MagicMock()
    scored_row.excellent = 5
    scored_row.good = 3
    scored_row.fair = 2
    scored_row.poor = 1

    scored_result = MagicMock()
    scored_result.one.return_value = scored_row

    unscored_result = MagicMock()
    unscored_result.scalar_one.return_value = 4

    total_result = MagicMock()
    total_result.scalar_one.return_value = 15

    db = _make_mock_db([scored_result, unscored_result, total_result])

    result = await score_distribution(db)

    assert result["excellent"] == 5
    assert result["good"] == 3
    assert result["fair"] == 2
    assert result["poor"] == 1
    assert result["unscored"] == 4
    assert result["total"] == 15


# ---------------------------------------------------------------------------
# regression_check tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regression_check_no_regression():
    """Should report no regression when scores are stable."""
    recent_result = MagicMock()
    recent_result.all.return_value = [(0.8,), (0.85,), (0.78,)]

    baseline_row = MagicMock()
    baseline_row.__getitem__ = lambda self, i: [0.80, 20][i]  # avg=0.80, count=20
    baseline_result = MagicMock()
    baseline_result.one.return_value = baseline_row

    db = _make_mock_db([recent_result, baseline_result])

    result = await regression_check(db, window=3, threshold=0.1)

    assert result["regression_detected"] is False
    assert result["recent_count"] == 3


@pytest.mark.asyncio
async def test_regression_check_detects_drop():
    """Should detect regression when recent scores drop significantly."""
    recent_result = MagicMock()
    recent_result.all.return_value = [(0.3,), (0.25,), (0.35,)]

    # Baseline avg is much higher
    baseline_row = MagicMock()
    baseline_row.__getitem__ = lambda self, i: [0.80, 50][i]
    baseline_result = MagicMock()
    baseline_result.one.return_value = baseline_row

    db = _make_mock_db([recent_result, baseline_result])

    result = await regression_check(db, window=3, threshold=0.1)

    assert result["regression_detected"] is True
    assert result["delta"] < -0.1


@pytest.mark.asyncio
async def test_regression_check_insufficient_data():
    """Should not flag regression with fewer than 2 scored reviews."""
    recent_result = MagicMock()
    recent_result.all.return_value = [(0.5,)]

    db = _make_mock_db([recent_result])

    result = await regression_check(db, window=10)

    assert result["regression_detected"] is False
    assert result["recent_avg"] is None


# ---------------------------------------------------------------------------
# full_report tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_report_structure():
    """Full report should contain all four sections."""
    # Mock all sub-queries
    db = AsyncMock()

    # We need multiple execute calls; simplest to mock at the report function level
    from unittest.mock import patch

    with (
        patch("app.eval.report.summary_by_node", return_value=[{"node_name": "review_code"}]),
        patch("app.eval.report.score_trend", return_value=[]),
        patch("app.eval.report.score_distribution", return_value={"total": 0}),
        patch("app.eval.report.regression_check", return_value={"regression_detected": False}),
    ):
        result = await full_report(db)

    assert "by_node" in result
    assert "trend_30d" in result
    assert "distribution" in result
    assert "regression" in result
