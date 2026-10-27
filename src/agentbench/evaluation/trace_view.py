"""A read-only reconstruction of tool calls from a run's trace events.

The evaluator only ever looks at the recorded trace, never at agent internals, so a
stored run can be scored again later and give the same result.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentbench.domain.enums import TraceEventType
from agentbench.domain.models import TokenUsage, TraceEvent

# Validation errors that mean the agent chose a tool it may not use, as opposed to
# choosing a valid tool and getting its arguments wrong.
SELECTION_ERRORS = frozenset({"UnknownToolError", "ToolNotAllowedError"})


@dataclass
class ToolCall:
    call_id: str
    tool: str
    requested_arguments: Any  # what the agent asked for (untrusted)
    validated_arguments: dict[str, Any] | None = None  # set only if validation passed
    validation_error: str | None = None  # error type, if validation failed
    attempts: int = 0
    failed_attempts: int = 0
    output: Any = None
    succeeded: bool = False

    @property
    def selection_error(self) -> bool:
        return self.validation_error in SELECTION_ERRORS

    @property
    def argument_error(self) -> bool:
        return self.validation_error is not None and not self.selection_error


@dataclass
class TraceView:
    calls: list[ToolCall] = field(default_factory=list)
    usages: list[TokenUsage | None] = field(default_factory=list)  # one per decision

    @classmethod
    def from_events(cls, events: Iterable[TraceEvent]) -> "TraceView":
        view = cls()
        by_id: dict[str, ToolCall] = {}
        for event in events:
            p = event.payload
            if event.type is TraceEventType.AGENT_DECISION:
                usage = p.get("usage")
                view.usages.append(TokenUsage.model_validate(usage) if usage else None)
                if p.get("kind") == "tool_call":
                    call = ToolCall(
                        call_id=f"call-{event.step}",
                        tool=p.get("tool"),
                        requested_arguments=p.get("arguments"),
                    )
                    by_id[call.call_id] = call
                    view.calls.append(call)
            elif event.type is TraceEventType.VALIDATION_ERROR:
                by_id[p["call_id"]].validation_error = p["error"]["type"]
            elif event.type is TraceEventType.TOOL_CALL_STARTED:
                call = by_id[p["call_id"]]
                call.attempts += 1
                call.validated_arguments = p.get("arguments")
            elif event.type is TraceEventType.TOOL_CALL_FAILED:
                by_id[p["call_id"]].failed_attempts += 1
            elif event.type is TraceEventType.TOOL_CALL_SUCCEEDED:
                call = by_id[p["call_id"]]
                call.succeeded = True
                call.output = p.get("output")
        return view

    @property
    def tool_attempts(self) -> int:
        return sum(c.attempts for c in self.calls)

    @property
    def failed_tool_attempts(self) -> int:
        return sum(c.failed_attempts for c in self.calls)

    def retrieved_record_ids(self) -> set[str]:
        """Record ids returned by successful calls. Uses the output conventions of the
        record tools: {"record": {...}} and {"records": [{...}, ...]}."""
        ids: set[str] = set()
        for call in self.calls:
            if not call.succeeded or not isinstance(call.output, dict):
                continue
            record = call.output.get("record")
            if isinstance(record, dict) and "id" in record:
                ids.add(record["id"])
            for item in call.output.get("records") or []:
                if isinstance(item, dict) and "id" in item:
                    ids.add(item["id"])
        return ids
