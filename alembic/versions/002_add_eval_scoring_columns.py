"""Add scoring columns to eval_logs — score_method, heuristic_scores, llm_judge_scores.

Revision ID: 002
Revises: 001
Create Date: 2026-03-30
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("eval_logs", sa.Column("score_method", sa.String(50), nullable=True))
    op.add_column("eval_logs", sa.Column("heuristic_scores", sa.JSON(), nullable=True))
    op.add_column("eval_logs", sa.Column("llm_judge_scores", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("eval_logs", "llm_judge_scores")
    op.drop_column("eval_logs", "heuristic_scores")
    op.drop_column("eval_logs", "score_method")
