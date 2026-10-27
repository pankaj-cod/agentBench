import uuid
from typing import Any

import pytest

from agentbench.domain.enums import TraceEventType
from agentbench.tracing.clock import FakeClock
from agentbench.tracing.recorder import TraceRecorder

RUN_ID = uuid.UUID(int=1)


def test_seq_is_contiguous_from_zero_and_ids_are_unique() -> None:
    recorder = TraceRecorder(RUN_ID, FakeClock())
    for step in range(5):
        recorder.record(TraceEventType.AGENT_DECISION, step)
    events = recorder.events()
    assert [e.seq for e in events] == [0, 1, 2, 3, 4]
    assert len({e.id for e in events}) == 5
    assert all(e.run_id == RUN_ID for e in events)


def test_elapsed_time_comes_from_the_monotonic_clock() -> None:
    clock = FakeClock()
    clock.advance(10)  # time before the recorder exists does not count
    recorder = TraceRecorder(RUN_ID, clock)
    first = recorder.record(TraceEventType.RUN_STARTED, 0)
    clock.advance(0.25)
    second = recorder.record(TraceEventType.RUN_FINISHED, 0)
    assert (first.elapsed_us, second.elapsed_us) == (0, 250_000)
    assert second.occurred_at > first.occurred_at


def test_payload_is_copied_so_later_mutation_cannot_change_the_trace() -> None:
    recorder = TraceRecorder(RUN_ID, FakeClock())
    payload: dict[str, Any] = {"items": [1, 2]}
    event = recorder.record(TraceEventType.AGENT_DECISION, 1, payload)
    payload["items"].append(3)
    assert event.payload == {"items": [1, 2]}


def test_non_json_payload_raises_instead_of_being_stringified() -> None:
    recorder = TraceRecorder(RUN_ID, FakeClock())
    with pytest.raises(TypeError):
        recorder.record(TraceEventType.AGENT_DECISION, 1, {"bad": object()})
    assert recorder.events() == ()


def test_oversized_payload_is_truncated_visibly() -> None:
    recorder = TraceRecorder(RUN_ID, FakeClock(), max_payload_bytes=100)
    event = recorder.record(TraceEventType.TOOL_CALL_SUCCEEDED, 1, {"output": "x" * 500})
    assert event.payload["truncated"] is True
    assert event.payload["original_bytes"] > 500
    assert len(event.payload["preview"]) <= 100


def test_redaction_hook_is_applied_before_storage() -> None:
    def redact(payload: dict[str, Any]) -> dict[str, Any]:
        return {k: ("***" if k == "api_key" else v) for k, v in payload.items()}

    recorder = TraceRecorder(RUN_ID, FakeClock(), redact=redact)
    event = recorder.record(TraceEventType.RUN_STARTED, 0, {"api_key": "secret", "n": 1})
    assert event.payload == {"api_key": "***", "n": 1}


def test_events_returns_a_snapshot_and_close_stops_recording() -> None:
    recorder = TraceRecorder(RUN_ID, FakeClock())
    recorder.record(TraceEventType.RUN_STARTED, 0)
    snapshot = recorder.events()
    recorder.record(TraceEventType.RUN_FINISHED, 0)
    assert len(snapshot) == 1
    assert len(recorder.close()) == 2
    with pytest.raises(RuntimeError, match="closed"):
        recorder.record(TraceEventType.RUN_FINISHED, 0)


def test_injected_id_factory_makes_event_ids_deterministic() -> None:
    ids = iter(uuid.UUID(int=n) for n in range(100, 200))
    recorder = TraceRecorder(RUN_ID, FakeClock(), id_factory=lambda: next(ids))
    assert recorder.record(TraceEventType.RUN_STARTED, 0).id == uuid.UUID(int=100)
