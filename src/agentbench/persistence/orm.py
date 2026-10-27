"""Database tables for agent runs and their trace events.

The database enforces the same rules as the domain models (status matches termination
reason, COMPLETED runs have an answer, gap-free unique seq per run), so a bug in Python
code cannot store a contradictory row.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from agentbench.domain.enums import AgentRunStatus, TerminationReason, TraceEventType

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def str_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """VARCHAR + CHECK constraint instead of a native PostgreSQL ENUM type, so adding a
    value later is a constraint change rather than an ALTER TYPE."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        validate_strings=True,
        values_callable=lambda members: [m.value for m in members],
    )


TIMESTAMPTZ = DateTime(timezone=True)


class AgentRunRow(Base):
    __tablename__ = "agent_run"
    __table_args__ = (
        CheckConstraint(
            "(status = 'COMPLETED' AND termination_reason = 'COMPLETED')"
            " OR (status = 'ERROR' AND termination_reason = 'HARNESS_ERROR')"
            " OR (status = 'FAILED' AND termination_reason IN ("
            "'MAX_STEPS_EXCEEDED', 'RUN_TIMEOUT', 'TOOL_FAILURE_EXHAUSTED', 'AGENT_ERROR'))",
            name="status_matches_reason",
        ),
        CheckConstraint(
            "status <> 'COMPLETED' OR final_answer IS NOT NULL", name="completed_has_answer"
        ),
        CheckConstraint(
            "step_count >= 0 AND tool_call_count >= 0 AND duration_us >= 0", name="nonnegative"
        ),
        CheckConstraint("finished_at >= started_at", name="time_order"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    task_key: Mapped[str] = mapped_column(String(100))
    task: Mapped[dict[str, Any]] = mapped_column(JSONB)  # snapshot of the Task as run
    limits: Mapped[dict[str, Any]] = mapped_column(JSONB)
    agent_name: Mapped[str] = mapped_column(String(100))
    agent_version: Mapped[str] = mapped_column(String(50))
    status: Mapped[AgentRunStatus] = mapped_column(str_enum(AgentRunStatus, "status"))
    termination_reason: Mapped[TerminationReason] = mapped_column(
        str_enum(TerminationReason, "termination_reason")
    )
    final_answer: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    step_count: Mapped[int] = mapped_column(Integer)
    tool_call_count: Mapped[int] = mapped_column(Integer)
    duration_us: Mapped[int] = mapped_column(BigInteger)
    started_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ)
    finished_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, server_default=func.now())

    events: Mapped[list["TraceEventRow"]] = relationship(
        order_by="TraceEventRow.seq", cascade="all, delete-orphan", passive_deletes=True
    )


class TraceEventRow(Base):
    __tablename__ = "trace_event"
    __table_args__ = (
        # Also the index used to read a run's trace in order.
        UniqueConstraint("agent_run_id", "seq"),
        CheckConstraint("seq >= 0 AND step >= 0 AND elapsed_us >= 0", name="nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)  # the event id
    agent_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_run.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer)
    step: Mapped[int] = mapped_column(Integer)
    type: Mapped[TraceEventType] = mapped_column(str_enum(TraceEventType, "type"))
    occurred_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ)
    elapsed_us: Mapped[int] = mapped_column(BigInteger)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
