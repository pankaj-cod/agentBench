import asyncio
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.domain.enums import TerminationReason
from agentbench.domain.models import RunLimits, RunResult, Task
from agentbench.persistence.orm import AgentRunRow, TraceEventRow
from agentbench.persistence.uow import UnitOfWork
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import DEFAULT_RECORDS, RecordStore, default_registry
from agentbench.tools.mock.calculator import CalculatorTool
from agentbench.tools.mock.faults import FaultInjectingTool
from agentbench.tools.mock.records import LookupRecordTool, SearchRecordsTool
from agentbench.tools.registry import ToolRegistry
from agentbench.tracing.clock import FakeClock

pytestmark = pytest.mark.postgres

T0 = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
ALL_TOOLS = ("search_records", "lookup_record", "calculator")


def run_task(prompt: str, registry: ToolRegistry | None = None) -> RunResult:
    runner = AgentRunner(registry or default_registry(), clock=FakeClock(T0))
    task = Task(key="persist", prompt=prompt, allowed_tools=ALL_TOOLS)
    return asyncio.run(runner.run(task, RuleBasedAgent(), RunLimits()))


def save(uow_factory: Callable[[], UnitOfWork], result: RunResult) -> None:
    with uow_factory() as uow:
        uow.agent_runs.add(result)
        uow.commit()


def load(uow_factory: Callable[[], UnitOfWork], run_id: uuid.UUID) -> RunResult | None:
    with uow_factory() as uow:
        return uow.agent_runs.get(run_id)


def test_successful_run_round_trips_exactly(uow_factory: Callable[[], UnitOfWork]) -> None:
    result = run_task("What is 17% of the quantity of record A-12?")
    save(uow_factory, result)
    loaded = load(uow_factory, result.run_id)
    assert loaded == result
    assert loaded is not None and loaded.final_answer == "20.4"
    assert len(loaded.events) == 10


def test_failed_run_keeps_its_error_and_full_trace(uow_factory: Callable[[], UnitOfWork]) -> None:
    store = RecordStore(DEFAULT_RECORDS)
    flaky = FaultInjectingTool(LookupRecordTool(store), failures=99, mode="transient")
    registry = ToolRegistry([SearchRecordsTool(store), flaky, CalculatorTool()])
    result = run_task("What is the quantity of record A-12?", registry)
    save(uow_factory, result)

    loaded = load(uow_factory, result.run_id)
    assert loaded == result
    assert loaded is not None
    assert loaded.termination_reason is TerminationReason.TOOL_FAILURE_EXHAUSTED
    assert loaded.final_answer is None
    assert "failed on all 3 attempts" in (loaded.error_message or "")
    failures = [e for e in loaded.events if e.type == "TOOL_CALL_FAILED"]
    assert [e.payload["error"]["retryable"] for e in failures] == [True, True, True]


def test_every_event_has_run_id_event_id_seq_type_timestamp_and_payload(
    uow_factory: Callable[[], UnitOfWork], session_factory: sessionmaker[Session]
) -> None:
    result = run_task("Find records matching 'cable'")
    save(uow_factory, result)
    with session_factory() as session:
        rows = session.scalars(
            select(TraceEventRow)
            .where(TraceEventRow.agent_run_id == result.run_id)
            .order_by(TraceEventRow.seq)
        ).all()
    assert [r.id for r in rows] == [e.id for e in result.events]
    assert [r.seq for r in rows] == list(range(len(rows)))
    for row in rows:
        assert row.agent_run_id == result.run_id
        assert row.occurred_at.tzinfo is not None
        assert isinstance(row.payload, dict)


def test_events_are_read_back_in_seq_order_regardless_of_insert_order(
    uow_factory: Callable[[], UnitOfWork], session_factory: sessionmaker[Session]
) -> None:
    result = run_task("What is the quantity of record A-12?")
    with session_factory() as session:
        session.add(_run_row(result.run_id))
        session.flush()
        for event in reversed(result.events):  # insert newest first
            session.add(
                TraceEventRow(
                    id=event.id,
                    agent_run_id=result.run_id,
                    seq=event.seq,
                    step=event.step,
                    type=event.type,
                    occurred_at=event.occurred_at,
                    elapsed_us=event.elapsed_us,
                    payload=event.payload,
                )
            )
        session.commit()
    with uow_factory() as uow:
        assert [e.seq for e in uow.agent_runs.events(result.run_id)] == list(
            range(len(result.events))
        )


def test_unknown_run_returns_none(uow_factory: Callable[[], UnitOfWork]) -> None:
    assert load(uow_factory, uuid.uuid4()) is None


def test_nothing_is_saved_without_commit_or_after_an_exception(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    result = run_task("What is the quantity of record A-12?")
    with uow_factory() as uow:
        uow.agent_runs.add(result)  # no commit
    assert load(uow_factory, result.run_id) is None

    with pytest.raises(RuntimeError), uow_factory() as uow:
        uow.agent_runs.add(result)
        raise RuntimeError("crash before commit")
    assert load(uow_factory, result.run_id) is None


def test_deleting_a_run_deletes_its_events(
    uow_factory: Callable[[], UnitOfWork], session_factory: sessionmaker[Session]
) -> None:
    result = run_task("What is the quantity of record A-12?")
    save(uow_factory, result)
    with session_factory() as session:
        session.execute(text("DELETE FROM agent_run WHERE id = :id"), {"id": result.run_id})
        remaining = session.scalar(
            select(func.count())
            .select_from(TraceEventRow)
            .where(TraceEventRow.agent_run_id == result.run_id)
        )
    assert remaining == 0


# --- Database constraints, tested with raw SQL so Python-side validation is bypassed ---

VALID_RUN: dict[str, Any] = {
    "task_key": "t",
    "task": "{}",
    "limits": "{}",
    "agent_name": "a",
    "agent_version": "1",
    "status": "COMPLETED",
    "termination_reason": "COMPLETED",
    "final_answer": "x",
    "error_message": None,
    "step_count": 1,
    "tool_call_count": 0,
    "duration_us": 0,
    "started_at": T0,
    "finished_at": T0,
}


def insert_run(session: Session, **overrides: Any) -> uuid.UUID:
    params = {"id": uuid.uuid4(), **VALID_RUN, **overrides}
    columns = ", ".join(params)
    values = ", ".join(
        f"CAST(:{k} AS jsonb)" if k in {"task", "limits"} else f":{k}" for k in params
    )
    session.execute(text(f"INSERT INTO agent_run ({columns}) VALUES ({values})"), params)
    return params["id"]


def insert_event(session: Session, run_id: uuid.UUID, **overrides: Any) -> None:
    params = {
        "id": uuid.uuid4(),
        "agent_run_id": run_id,
        "seq": 0,
        "step": 0,
        "type": "RUN_STARTED",
        "occurred_at": T0,
        "elapsed_us": 0,
        "payload": json.dumps({}),
        **overrides,
    }
    session.execute(
        text(
            "INSERT INTO trace_event (id, agent_run_id, seq, step, type, occurred_at, "
            "elapsed_us, payload) VALUES (:id, :agent_run_id, :seq, :step, :type, "
            ":occurred_at, :elapsed_us, CAST(:payload AS jsonb))"
        ),
        params,
    )


def test_valid_raw_rows_are_accepted(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = insert_run(session)
        insert_event(session, run_id)
        session.flush()


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        ({"termination_reason": "MAX_STEPS_EXCEEDED"}, "ck_agent_run_status_matches_reason"),
        (
            {"status": "ERROR", "termination_reason": "AGENT_ERROR"},
            "ck_agent_run_status_matches_reason",
        ),
        ({"final_answer": None}, "ck_agent_run_completed_has_answer"),
        ({"step_count": -1}, "ck_agent_run_nonnegative"),
        ({"finished_at": T0 - timedelta(seconds=1)}, "ck_agent_run_time_order"),
        # Added in migration 0002 by hand; autogenerate would have missed it.
        ({"input_tokens": -1}, "ck_agent_run_tokens_nonnegative"),
        ({"output_tokens": -5}, "ck_agent_run_tokens_nonnegative"),
        # An unknown enum value also breaks status_matches_reason. PostgreSQL checks
        # CHECK constraints in alphabetical order of name and reports only the first
        # failure, so either name proves the row was rejected.
        ({"status": "DONE"}, r"ck_agent_run_status\b|ck_agent_run_status_matches_reason"),
        (
            {"termination_reason": "GAVE_UP"},
            "ck_agent_run_termination_reason|ck_agent_run_status_matches_reason",
        ),
    ],
)
def test_agent_run_constraints(
    session_factory: sessionmaker[Session], overrides: dict[str, Any], constraint: str
) -> None:
    with session_factory() as session, pytest.raises(IntegrityError, match=constraint):
        insert_run(session, **overrides)


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        ({"seq": -1}, "ck_trace_event_nonnegative"),
        ({"type": "SOMETHING_ELSE"}, "ck_trace_event_type"),
        ({"agent_run_id": uuid.uuid4()}, "fk_trace_event_agent_run_id_agent_run"),
    ],
)
def test_trace_event_constraints(
    session_factory: sessionmaker[Session], overrides: dict[str, Any], constraint: str
) -> None:
    with session_factory() as session:
        run_id = insert_run(session)
        with pytest.raises(IntegrityError, match=constraint):
            insert_event(session, run_id, **overrides)


def test_seq_is_unique_within_a_run(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = insert_run(session)
        insert_event(session, run_id, seq=0)
        with pytest.raises(IntegrityError, match="uq_trace_event_agent_run_id_seq"):
            insert_event(session, run_id, seq=0)


def test_same_seq_is_allowed_in_different_runs(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        first, second = insert_run(session), insert_run(session)
        insert_event(session, first, seq=0)
        insert_event(session, second, seq=0)
        session.flush()


def _run_row(run_id: uuid.UUID) -> AgentRunRow:
    return AgentRunRow(
        id=run_id,
        **{k: v for k, v in VALID_RUN.items() if k not in {"task", "limits"}},
        task=Task(key="persist", prompt="x").model_dump(mode="json"),
        limits=RunLimits().model_dump(mode="json"),
    )
