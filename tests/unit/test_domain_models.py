import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from agentbench.domain.enums import AgentRunStatus, TerminationReason, TraceEventType
from agentbench.domain.models import (
    Decision,
    FinalAnswerDecision,
    Observation,
    RunLimits,
    RunResult,
    Task,
    ToolCallDecision,
    ToolErrorInfo,
    TraceEvent,
)

T0 = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        (TerminationReason.COMPLETED, AgentRunStatus.COMPLETED),
        (TerminationReason.MAX_STEPS_EXCEEDED, AgentRunStatus.FAILED),
        (TerminationReason.RUN_TIMEOUT, AgentRunStatus.FAILED),
        (TerminationReason.TOOL_FAILURE_EXHAUSTED, AgentRunStatus.FAILED),
        (TerminationReason.AGENT_ERROR, AgentRunStatus.FAILED),
        (TerminationReason.HARNESS_ERROR, AgentRunStatus.ERROR),
    ],
)
def test_every_termination_reason_maps_to_one_status(
    reason: TerminationReason, status: AgentRunStatus
) -> None:
    assert reason.status is status


@pytest.mark.parametrize(
    "bad",
    [{"max_steps": 0}, {"max_retries": -1}, {"tool_timeout_s": 0}, {"unknown": 1}],
)
def test_run_limits_reject_out_of_range_and_unknown_fields(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunLimits.model_validate(bad)


def test_task_key_must_be_a_slug() -> None:
    with pytest.raises(ValidationError):
        Task(key="Not A Slug", prompt="x")


def test_decision_union_is_selected_by_kind() -> None:
    adapter = TypeAdapter(Decision)
    call = adapter.validate_python({"kind": "tool_call", "tool": "calculator"})
    answer = adapter.validate_python({"kind": "final_answer", "answer": "42"})
    assert isinstance(call, ToolCallDecision)
    assert isinstance(answer, FinalAnswerDecision)


def test_observation_requires_exactly_one_of_output_or_error() -> None:
    error = ToolErrorInfo(type="X", message="m", retryable=False)
    with pytest.raises(ValidationError):
        Observation(call_id="c", tool="t", ok=True, error=error)
    with pytest.raises(ValidationError):
        Observation(call_id="c", tool="t", ok=False)


def _event(run_id: uuid.UUID, seq: int) -> TraceEvent:
    return TraceEvent(
        id=uuid.uuid4(),
        run_id=run_id,
        seq=seq,
        step=0,
        type=TraceEventType.RUN_STARTED,
        occurred_at=T0,
        elapsed_us=0,
        payload={},
    )


def _result(**overrides: object) -> RunResult:
    run_id = uuid.uuid4()
    fields: dict[str, object] = {
        "run_id": run_id,
        "task": Task(key="t", prompt="p"),
        "limits": RunLimits(),
        "agent_name": "scripted",
        "agent_version": "1",
        "status": AgentRunStatus.COMPLETED,
        "termination_reason": TerminationReason.COMPLETED,
        "final_answer": "42",
        "error_message": None,
        "step_count": 1,
        "tool_call_count": 0,
        "started_at": T0,
        "finished_at": T0 + timedelta(seconds=1),
        "duration_us": 1_000_000,
        "events": (_event(run_id, 0), _event(run_id, 1)),
    }
    fields.update(overrides)
    return RunResult.model_validate(fields)


def test_valid_run_result() -> None:
    assert _result().final_answer == "42"


def test_run_result_rejects_status_that_contradicts_reason() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        _result(termination_reason=TerminationReason.MAX_STEPS_EXCEEDED)


def test_completed_run_result_requires_final_answer() -> None:
    with pytest.raises(ValidationError, match="final answer"):
        _result(final_answer=None)


def test_run_result_rejects_gaps_in_event_sequence() -> None:
    run_id = uuid.uuid4()
    with pytest.raises(ValidationError, match="seq"):
        _result(run_id=run_id, events=(_event(run_id, 0), _event(run_id, 2)))


def test_run_result_rejects_events_from_another_run() -> None:
    with pytest.raises(ValidationError, match="belong"):
        _result(events=(_event(uuid.uuid4(), 0),))


def test_run_result_rejects_naive_timestamps() -> None:
    with pytest.raises(ValidationError):
        _result(started_at=datetime(2026, 10, 27, 12, 0))
