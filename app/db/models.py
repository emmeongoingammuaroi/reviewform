"""Database models for review sessions and eval logs."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ReviewSession(Base):
    """Tracks each code review request and its result."""

    __tablename__ = "review_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    input_type: Mapped[str] = mapped_column(String(50))  # "snippet" or "github_pr"
    input_content: Mapped[str] = mapped_column(Text)  # raw code or PR URL
    review_output: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending/reviewing/done/error
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class EvalLog(Base):
    """Logs prompt/response pairs with quality scores for evaluation."""

    __tablename__ = "eval_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    node_name: Mapped[str] = mapped_column(String(100))  # which LangGraph node
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)  # 0.0-1.0, null until scored
    score_reason: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
