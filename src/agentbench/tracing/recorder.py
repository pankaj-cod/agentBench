import json
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from agentbench.domain.enums import TraceEventType
from agentbench.domain.models import TraceEvent
from agentbench.tracing.clock import Clock

DEFAULT_MAX_PAYLOAD_BYTES = 16 * 1024

Redactor = Callable[[dict[str, Any]], dict[str, Any]]


def _no_redaction(payload: dict[str, Any]) -> dict[str, Any]:
    return payload


class TraceRecorder:
    """Builds the ordered, append-only event list for one run.

    The recorder is the only writer for its run, so `seq` is simply the number of
    events recorded so far: 0, 1, 2, ... with no gaps. Order never depends on
    timestamps. Payloads are copied through JSON, so later changes to the caller's
    objects cannot alter what was recorded, and oversized payloads are truncated
    visibly rather than silently.
    """

    def __init__(
        self,
        run_id: uuid.UUID,
        clock: Clock,
        *,
        id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
        redact: Redactor = _no_redaction,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self._run_id = run_id
        self._clock = clock
        self._id_factory = id_factory
        self._redact = redact
        self._max_payload_bytes = max_payload_bytes
        self._start_us = clock.monotonic_us()
        self._events: list[TraceEvent] = []
        self._closed = False

    @property
    def run_id(self) -> uuid.UUID:
        return self._run_id

    def record(
        self, type: TraceEventType, step: int, payload: Mapping[str, Any] | None = None
    ) -> TraceEvent:
        if self._closed:
            raise RuntimeError("cannot record events after the recorder is closed")
        event = TraceEvent(
            id=self._id_factory(),
            run_id=self._run_id,
            seq=len(self._events),
            step=step,
            type=type,
            occurred_at=self._clock.now(),
            elapsed_us=self._clock.monotonic_us() - self._start_us,
            payload=self._prepare(payload or {}),
        )
        self._events.append(event)
        return event

    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def close(self) -> tuple[TraceEvent, ...]:
        self._closed = True
        return self.events()

    def _prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        redacted = self._redact(dict(payload))
        # No `default=`: a payload that is not plain JSON is a bug in AgentBench and must
        # raise, not be silently turned into a string.
        text = json.dumps(redacted, sort_keys=True, allow_nan=False)
        size = len(text.encode())
        if size > self._max_payload_bytes:
            preview = text.encode()[: self._max_payload_bytes].decode(errors="ignore")
            return {"truncated": True, "original_bytes": size, "preview": preview}
        return json.loads(text)
