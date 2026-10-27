import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentbench.agents.base import Agent, AgentContext
from agentbench.domain.agent_config import tool_schemas_hash
from agentbench.domain.enums import ExecutionMode, TerminationReason, TraceEventType
from agentbench.domain.errors import (
    AgentInfrastructureError,
    InvalidTaskError,
    ToolError,
    ToolValidationError,
)
from agentbench.domain.models import (
    FinalAnswerDecision,
    HistoryStep,
    Observation,
    RunLimits,
    RunResult,
    Task,
    ToolCallDecision,
)
from agentbench.tools.registry import ToolRegistry, ValidatedToolCall
from agentbench.tracing.clock import Clock, SystemClock
from agentbench.tracing.recorder import TraceRecorder


@dataclass(frozen=True)
class _Stop:
    """Why the loop ended. Converted into the RunResult."""

    reason: TerminationReason
    final_answer: str | None = None
    error_message: str | None = None


@dataclass
class _Counters:
    step_count: int = 0  # agent decisions requested
    tool_call_count: int = 0  # tool-call decisions, valid or not; retries are not counted
    decisions: int = 0  # decisions actually received
    decisions_with_usage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_schemas_hash: str | None = None

    def add_usage(self, decision: ToolCallDecision | FinalAnswerDecision) -> None:
        self.decisions += 1
        if decision.usage is not None:
            self.decisions_with_usage += 1
            self.input_tokens += decision.usage.input_tokens
            self.output_tokens += decision.usage.output_tokens

    def token_totals(self) -> tuple[int | None, int | None]:
        """Totals only when every decision reported usage; a partial sum would
        understate usage, so it is reported as unavailable instead."""
        if self.decisions == 0 or self.decisions_with_usage < self.decisions:
            return None, None
        return self.input_tokens, self.output_tokens


class AgentRunner:
    """Runs one task with one agent: decide -> validate -> execute -> observe, repeated.

    Guarantees:
    - `run` always returns a RunResult for agent and tool failures; it does not raise.
    - Every run ends with exactly one TerminationReason, recorded in RUN_FINISHED.
    - A tool result reaches the agent only if the tool actually succeeded.
    - Limits: `max_steps` decisions, a whole-run deadline, a per-call timeout, and at
      most `max_retries` retries of retryable tool errors, with exponential backoff.
    - An unexpected exception in AgentBench's own code becomes HARNESS_ERROR.
    - An AgentInfrastructureError from the agent (e.g. the model provider stayed
      unavailable after retries) becomes MODEL_UNAVAILABLE, not AGENT_ERROR.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        clock: Clock | None = None,
        id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        self._registry = registry
        self._clock = clock or SystemClock()
        self._id_factory = id_factory

    async def run(
        self,
        task: Task,
        agent: Agent,
        limits: RunLimits | None = None,
        *,
        execution_mode: ExecutionMode = ExecutionMode.LIVE,
        source_run_id: uuid.UUID | None = None,
    ) -> RunResult:
        limits = limits or RunLimits()
        unknown = set(task.allowed_tools) - self._registry.names()
        if unknown:
            # A configuration error, not a run outcome: no run is started.
            raise InvalidTaskError(f"task {task.key} allows unregistered tools: {sorted(unknown)}")

        run_id = self._id_factory()
        recorder = TraceRecorder(run_id, self._clock, id_factory=self._id_factory)
        started_at = self._clock.now()
        start_us = self._clock.monotonic_us()
        deadline_us = start_us + round(limits.run_timeout_s * 1_000_000)
        counters = _Counters()

        try:
            stop = await self._loop(
                task, agent, limits, recorder, counters, deadline_us, execution_mode, source_run_id
            )
        except Exception as exc:
            stop = _Stop(TerminationReason.HARNESS_ERROR, error_message=_describe(exc))

        input_tokens, output_tokens = counters.token_totals()
        recorder.record(
            TraceEventType.RUN_FINISHED,
            counters.step_count,
            {
                "status": stop.reason.status,
                "termination_reason": stop.reason,
                "final_answer": stop.final_answer,
                "error_message": stop.error_message,
                "step_count": counters.step_count,
                "tool_call_count": counters.tool_call_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tool_schemas_hash": counters.tool_schemas_hash,
            },
        )
        return RunResult(
            run_id=run_id,
            task=task,
            limits=limits,
            agent_name=agent.name,
            agent_version=agent.version,
            status=stop.reason.status,
            termination_reason=stop.reason,
            final_answer=stop.final_answer,
            error_message=stop.error_message,
            step_count=counters.step_count,
            tool_call_count=counters.tool_call_count,
            started_at=started_at,
            finished_at=self._clock.now(),
            duration_us=self._clock.monotonic_us() - start_us,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agent_config_hash=getattr(agent, "config_hash", None),
            tool_schemas_hash=counters.tool_schemas_hash,
            execution_mode=execution_mode,
            source_run_id=source_run_id,
            events=recorder.close(),
        )

    async def _loop(
        self,
        task: Task,
        agent: Agent,
        limits: RunLimits,
        recorder: TraceRecorder,
        counters: _Counters,
        deadline_us: int,
        execution_mode: ExecutionMode,
        source_run_id: uuid.UUID | None,
    ) -> _Stop:
        allowed = frozenset(task.allowed_tools)
        describe = getattr(agent, "describe", None)
        recorder.record(
            TraceEventType.RUN_STARTED,
            0,
            {
                "task": task.model_dump(mode="json"),
                "limits": limits.model_dump(mode="json"),
                "agent": {
                    "name": agent.name,
                    "version": agent.version,
                    "config": describe() if describe else None,
                },
                "execution_mode": execution_mode,
                "source_run_id": str(source_run_id) if source_run_id else None,
            },
        )
        tool_schemas = tuple(self._registry.schemas(allowed))
        counters.tool_schemas_hash = tool_schemas_hash(tool_schemas)
        history: list[HistoryStep] = []

        for step in range(1, limits.max_steps + 1):
            remaining_s = self._remaining_s(deadline_us)
            if remaining_s <= 0:
                return _Stop(TerminationReason.RUN_TIMEOUT, error_message=_deadline_message(step))
            counters.step_count = step
            context = AgentContext(task=task, step=step, tools=tool_schemas, history=tuple(history))

            try:
                decision = await asyncio.wait_for(agent.decide(context), timeout=remaining_s)
            except TimeoutError:
                return _Stop(
                    TerminationReason.RUN_TIMEOUT,
                    error_message=f"agent did not decide within the remaining {remaining_s:.3f}s",
                )
            except AgentInfrastructureError as exc:
                return _Stop(TerminationReason.MODEL_UNAVAILABLE, error_message=_describe(exc))
            except Exception as exc:
                return _Stop(TerminationReason.AGENT_ERROR, error_message=_describe(exc))

            payload = _decision_payload(decision)
            if isinstance(payload, str):
                return _Stop(TerminationReason.AGENT_ERROR, error_message=payload)
            recorder.record(TraceEventType.AGENT_DECISION, step, payload)
            counters.add_usage(decision)

            if isinstance(decision, FinalAnswerDecision):
                recorder.record(TraceEventType.FINAL_ANSWER, step, {"answer": decision.answer})
                return _Stop(TerminationReason.COMPLETED, final_answer=decision.answer)

            counters.tool_call_count += 1
            call_id = f"call-{step}"
            try:
                call = self._registry.validate(call_id, decision.tool, decision.arguments, allowed)
            except ToolValidationError as err:
                info = err.to_info()
                recorder.record(
                    TraceEventType.VALIDATION_ERROR,
                    step,
                    {"call_id": call_id, "tool": decision.tool, "error": info.model_dump()},
                )
                observation = Observation(call_id=call_id, tool=decision.tool, ok=False, error=info)
            else:
                result = await self._execute(call, step, limits, recorder, deadline_us)
                if isinstance(result, _Stop):
                    return result
                observation = result
            history.append(HistoryStep(step=step, decision=decision, observation=observation))

        return _Stop(
            TerminationReason.MAX_STEPS_EXCEEDED,
            error_message=f"no final answer after {limits.max_steps} steps",
        )

    async def _execute(
        self,
        call: ValidatedToolCall,
        step: int,
        limits: RunLimits,
        recorder: TraceRecorder,
        deadline_us: int,
    ) -> Observation | _Stop:
        tool = call.tool.name
        arguments = call.args.model_dump(mode="json")
        for attempt in range(1, limits.max_retries + 2):
            remaining_s = self._remaining_s(deadline_us)
            if remaining_s <= 0:
                return _Stop(TerminationReason.RUN_TIMEOUT, error_message=_deadline_message(step))
            timeout_s = min(limits.tool_timeout_s, remaining_s)
            base = {"call_id": call.call_id, "tool": tool, "attempt": attempt}
            recorder.record(
                TraceEventType.TOOL_CALL_STARTED,
                step,
                {**base, "arguments": arguments, "timeout_s": timeout_s},
            )
            started_us = self._clock.monotonic_us()
            try:
                output = await self._registry.execute(call, timeout_s)
            except ToolError as err:
                info = err.to_info()
                recorder.record(
                    TraceEventType.TOOL_CALL_FAILED,
                    step,
                    {
                        **base,
                        "duration_us": self._clock.monotonic_us() - started_us,
                        "error": info.model_dump(),
                    },
                )
                if not err.retryable:
                    return Observation(call_id=call.call_id, tool=tool, ok=False, error=info)
                if attempt > limits.max_retries:
                    return _Stop(
                        TerminationReason.TOOL_FAILURE_EXHAUSTED,
                        error_message=(
                            f"{tool} failed on all {attempt} attempts; "
                            f"last error: {info.type}: {info.message}"
                        ),
                    )
                await self._clock.sleep(limits.retry_backoff_s * 2 ** (attempt - 1))
                continue
            recorder.record(
                TraceEventType.TOOL_CALL_SUCCEEDED,
                step,
                {**base, "duration_us": self._clock.monotonic_us() - started_us, "output": output},
            )
            return Observation(call_id=call.call_id, tool=tool, ok=True, output=output)
        raise AssertionError("unreachable: the last attempt always returns")

    def _remaining_s(self, deadline_us: int) -> float:
        return (deadline_us - self._clock.monotonic_us()) / 1_000_000


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _deadline_message(step: int) -> str:
    return f"run deadline reached at step {step}"


def _decision_payload(decision: object) -> dict[str, Any] | str:
    """The decision as a JSON payload, or an error message if the agent returned
    something that is not a valid, serialisable decision."""
    if not isinstance(decision, ToolCallDecision | FinalAnswerDecision):
        return f"agent returned {type(decision).__name__}, not a decision"
    try:
        payload = decision.model_dump(mode="json")
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return f"agent decision is not valid JSON: {_describe(exc)}"
    return payload
