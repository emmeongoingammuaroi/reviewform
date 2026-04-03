"""Review API routes — the main entry point for code reviews."""

import asyncio
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.builder import review_graph
from app.api.schemas.review import HumanFeedback, ReviewRequest, ReviewResponse
from app.core.config import settings
from app.core.dependencies import get_current_user
from app.db.base import get_db
from app.db.models import ReviewSession

router = APIRouter(prefix="/reviews", tags=["reviews"])
logger = structlog.get_logger(__name__)


@router.post("/", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
async def create_review(
    request: ReviewRequest,
    user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ReviewResponse:
    """Start a new code review.

    1. Creates a ReviewSession in the database
    2. Invokes the LangGraph review workflow
    3. The graph runs: fetch_diff -> retrieve_standards -> review_code -> format_response
    4. Pauses at the human_review node (interrupt_before)
    5. Returns the AI review for human approval
    """
    session_id = str(uuid.uuid4())

    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        user_id=user["user_id"],
    )

    logger.info("review.create", input_type=request.input_type)

    # Save session to database
    db_session = ReviewSession(
        id=uuid.UUID(session_id),
        user_id=user["user_id"],
        input_type=request.input_type.value,
        input_content=request.content,
        status="reviewing",
    )
    db.add(db_session)
    await db.commit()

    # Run the LangGraph workflow
    initial_state = {
        "session_id": session_id,
        "input_type": request.input_type.value,
        "raw_input": request.content,
        "language": request.language,
    }

    # thread_id enables checkpointing for human-in-the-loop
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await asyncio.wait_for(
            review_graph.ainvoke(initial_state, config=config),
            timeout=settings.review_timeout_seconds,
        )
    except TimeoutError:
        logger.error("review.timeout", session_id=session_id)
        db_session.status = "error"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Review timed out — try a smaller code snippet",
        )
    except Exception as e:
        logger.error("review.failed", error=str(e))
        db_session.status = "error"
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Review failed: {e}")

    # Update session with results
    db_session.review_output = result.get("summary", "")
    db_session.status = "awaiting_feedback"
    await db.commit()

    logger.info("review.completed", num_issues=len(result.get("issues", [])))

    return ReviewResponse(
        session_id=uuid.UUID(session_id),
        status="awaiting_feedback",
        summary=result.get("summary", ""),
        issues=result.get("issues", []),
        standards_used=result.get("standards", []),
        created_at=db_session.created_at,
    )


@router.post("/{session_id}/feedback")
async def submit_feedback(
    session_id: uuid.UUID,
    feedback: HumanFeedback,
    user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Submit human feedback on a review (human-in-the-loop).

    This resumes the LangGraph workflow from the human_review checkpoint.
    If approved, the review is finalized. If rejected, the code is re-reviewed
    with the human's comments as additional context.
    """
    logger.info("review.feedback", session_id=str(session_id), approved=feedback.approved)

    config = {"configurable": {"thread_id": str(session_id)}}

    # Resume the graph with human feedback injected into state
    result = await review_graph.ainvoke(
        {
            "human_approved": feedback.approved,
            "human_comments": feedback.comments,
        },
        config=config,
    )

    return {
        "session_id": session_id,
        "status": "approved" if feedback.approved else "re-reviewed",
        "summary": result.get("summary", ""),
        "issues": result.get("issues", []),
    }
