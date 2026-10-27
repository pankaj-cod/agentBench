import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentbench.domain.models import RunLimits, RunResult, Task, TraceEvent
from agentbench.persistence.orm import AgentRunRow, TraceEventRow


class AgentRunRepository:
    """Saves and loads agent runs with their traces.

    Converts between ORM rows and domain models in both directions, so ORM objects
    never leave this module. Never commits: the UnitOfWork owns the transaction.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, result: RunResult) -> None:
        row = AgentRunRow(
            id=result.run_id,
            task_key=result.task.key,
            task=result.task.model_dump(mode="json"),
            limits=result.limits.model_dump(mode="json"),
            agent_name=result.agent_name,
            agent_version=result.agent_version,
            status=result.status,
            termination_reason=result.termination_reason,
            final_answer=result.final_answer,
            error_message=result.error_message,
            step_count=result.step_count,
            tool_call_count=result.tool_call_count,
            duration_us=result.duration_us,
            started_at=result.started_at,
            finished_at=result.finished_at,
            events=[
                TraceEventRow(
                    id=e.id,
                    seq=e.seq,
                    step=e.step,
                    type=e.type,
                    occurred_at=e.occurred_at,
                    elapsed_us=e.elapsed_us,
                    payload=e.payload,
                )
                for e in result.events
            ],
        )
        self._session.add(row)

    def get(self, run_id: uuid.UUID) -> RunResult | None:
        row = self._session.get(AgentRunRow, run_id)
        if row is None:
            return None
        return RunResult(
            run_id=row.id,
            task=Task.model_validate(row.task),
            limits=RunLimits.model_validate(row.limits),
            agent_name=row.agent_name,
            agent_version=row.agent_version,
            status=row.status,
            termination_reason=row.termination_reason,
            final_answer=row.final_answer,
            error_message=row.error_message,
            step_count=row.step_count,
            tool_call_count=row.tool_call_count,
            started_at=row.started_at,
            finished_at=row.finished_at,
            duration_us=row.duration_us,
            events=tuple(self.events(run_id)),
        )

    def events(self, run_id: uuid.UUID) -> list[TraceEvent]:
        rows = self._session.scalars(
            select(TraceEventRow)
            .where(TraceEventRow.agent_run_id == run_id)
            .order_by(TraceEventRow.seq)
        )
        return [
            TraceEvent(
                id=r.id,
                run_id=r.agent_run_id,
                seq=r.seq,
                step=r.step,
                type=r.type,
                occurred_at=r.occurred_at,
                elapsed_us=r.elapsed_us,
                payload=r.payload,
            )
            for r in rows
        ]
