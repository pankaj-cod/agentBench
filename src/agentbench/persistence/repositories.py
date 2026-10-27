import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentbench.domain.benchmark import Expectations
from agentbench.domain.errors import AgentBenchError
from agentbench.domain.evaluation import CheckResult, Evaluation, RunMetrics
from agentbench.domain.models import RunLimits, RunResult, Task, TraceEvent
from agentbench.persistence.orm import AgentRunRow, EvaluationResultRow, TraceEventRow


class EvaluationConflictError(AgentBenchError):
    """A different evaluation is already stored for the same run and evaluator version:
    either the evaluator is not deterministic or the criteria changed without a new
    evaluator version. Both must be fixed, not overwritten."""


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
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
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
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
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


class EvaluationRepository:
    """Stores evaluations. Never commits; the UnitOfWork owns the transaction."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, evaluation: Evaluation) -> bool:
        """Store the evaluation unless an identical one exists. Returns True if a row was
        added. Raises EvaluationConflictError if a *different* evaluation is stored for
        the same run and evaluator version."""
        existing = self.get(evaluation.run_id, evaluation.evaluator_version)
        if existing is None:
            self._session.add(self._to_row(evaluation))
            return True
        if existing != evaluation:
            raise EvaluationConflictError(
                f"run {evaluation.run_id} already has a different {evaluation.evaluator_version}"
                " evaluation"
            )
        return False

    def get(self, run_id: uuid.UUID, evaluator_version: str) -> Evaluation | None:
        row = self._session.scalars(
            select(EvaluationResultRow).where(
                EvaluationResultRow.agent_run_id == run_id,
                EvaluationResultRow.evaluator_version == evaluator_version,
            )
        ).one_or_none()
        return None if row is None else self._to_domain(row)

    def list_for_run(self, run_id: uuid.UUID) -> list[Evaluation]:
        rows = self._session.scalars(
            select(EvaluationResultRow)
            .where(EvaluationResultRow.agent_run_id == run_id)
            .order_by(EvaluationResultRow.evaluator_version)
        )
        return [self._to_domain(r) for r in rows]

    @staticmethod
    def _to_row(e: Evaluation) -> EvaluationResultRow:
        return EvaluationResultRow(
            id=uuid.uuid4(),
            agent_run_id=e.run_id,
            evaluator_version=e.evaluator_version,
            pricing_version=e.pricing_version,
            case_key=e.case_key,
            outcome=e.outcome,
            failure_category=e.failure_category,
            failure_source=e.failure_source,
            explanation=e.explanation,
            checks=[c.model_dump(mode="json") for c in e.checks],
            metrics=e.metrics.model_dump(mode="json"),
            expectations=e.expectations.model_dump(mode="json"),
        )

    @staticmethod
    def _to_domain(row: EvaluationResultRow) -> Evaluation:
        return Evaluation(
            run_id=row.agent_run_id,
            case_key=row.case_key,
            evaluator_version=row.evaluator_version,
            pricing_version=row.pricing_version,
            outcome=row.outcome,
            failure_category=row.failure_category,
            failure_source=row.failure_source,
            explanation=row.explanation,
            checks=tuple(CheckResult.model_validate(c) for c in row.checks),
            metrics=RunMetrics.model_validate(row.metrics),
            expectations=Expectations.model_validate(row.expectations),
        )
