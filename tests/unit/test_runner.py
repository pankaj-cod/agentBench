import asyncio
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from agentbench.agents.base import AgentContext
from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.agents.scripted import ScriptedAgent
from agentbench.domain.enums import AgentRunStatus, TerminationReason
from agentbench.domain.errors import InvalidTaskError
from agentbench.domain.models import (
    FinalAnswerDecision,
    RunLimits,
    RunResult,
    Task,
    TokenUsage,
    ToolCallDecision,
)
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import default_registry
from agentbench.tools.registry import ToolRegistry
from agentbench.tracing.clock import FakeClock

TASK = Task(
    key="lookup",
    prompt="What is the quantity of record A-12?",
    allowed_tools=("lookup_record", "calculator"),
)


def counter_ids() -> Iterator[uuid.UUID]:
    n = 0
    while True:
        n += 1
        yield uuid.UUID(int=n)


def run(agent: Any, task: Task = TASK, registry: ToolRegistry | None = None) -> RunResult:
    runner = AgentRunner(registry or default_registry(), clock=FakeClock())
    return asyncio.run(runner.run(task, agent, RunLimits()))


class RecordingAgent:
    """Scripted agent that also keeps every context it was given."""

    name, version = "recording", "1"

    def __init__(self, decisions: list[ToolCallDecision | FinalAnswerDecision]) -> None:
        self.inner = ScriptedAgent(decisions)
        self.contexts: list[AgentContext] = []

    async def decide(self, context: AgentContext) -> ToolCallDecision | FinalAnswerDecision:
        self.contexts.append(context)
        return await self.inner.decide(context)


def test_task_that_allows_an_unregistered_tool_is_rejected_before_running() -> None:
    task = Task(key="bad", prompt="x", allowed_tools=("rm_rf",))
    with pytest.raises(InvalidTaskError, match="rm_rf"):
        run(RuleBasedAgent(), task)


def test_agent_sees_only_schemas_of_allowed_tools() -> None:
    agent = RecordingAgent([FinalAnswerDecision(answer="x")])
    run(agent)
    assert [t["name"] for t in agent.contexts[0].tools] == ["calculator", "lookup_record"]


def test_validation_error_is_returned_to_the_agent_with_field_details() -> None:
    agent = RecordingAgent(
        [
            ToolCallDecision(tool="lookup_record", arguments={"id": "A-12"}),
            FinalAnswerDecision(answer="done"),
        ]
    )
    run(agent)
    observation = agent.contexts[1].history[0].observation
    assert observation is not None and not observation.ok
    assert observation.error is not None
    assert observation.error.type == "ToolArgumentsError"
    assert observation.error.details["errors"][0]["loc"] == ["record_id"]


def test_successful_output_is_returned_to_the_agent() -> None:
    agent = RecordingAgent(
        [
            ToolCallDecision(tool="lookup_record", arguments={"record_id": "A-12"}),
            FinalAnswerDecision(answer="done"),
        ]
    )
    run(agent)
    observation = agent.contexts[1].history[0].observation
    assert observation is not None and observation.ok
    assert observation.output["record"]["quantity"] == 120


class ReturnsGarbage:
    name, version = "garbage", "1"

    async def decide(self, context: AgentContext) -> Any:
        return {"tool": "lookup_record"}  # a dict, not a Decision


class NonJsonArguments:
    name, version = "non-json", "1"

    async def decide(self, context: AgentContext) -> ToolCallDecision:
        return ToolCallDecision(tool="lookup_record", arguments={"record_id": object()})


@pytest.mark.parametrize("agent", [ReturnsGarbage(), NonJsonArguments()])
def test_invalid_decisions_are_agent_errors(agent: Any) -> None:
    result = run(agent)
    assert result.termination_reason is TerminationReason.AGENT_ERROR
    assert result.status is AgentRunStatus.FAILED
    assert result.final_answer is None


class BrokenRegistry(ToolRegistry):
    def schemas(self, names: Any) -> list[dict[str, Any]]:
        raise RuntimeError("bug inside AgentBench")


def test_bug_in_agentbench_becomes_harness_error_not_a_crash() -> None:
    registry = BrokenRegistry(default_registry()._tools.values())
    result = run(RuleBasedAgent(), registry=registry)
    assert result.termination_reason is TerminationReason.HARNESS_ERROR
    assert result.status is AgentRunStatus.ERROR
    assert result.error_message == "RuntimeError: bug inside AgentBench"
    assert [e.type for e in result.events] == ["RUN_STARTED", "RUN_FINISHED"]


def test_exponential_backoff_between_retries() -> None:
    from agentbench.tools.mock.calculator import CalculatorTool
    from agentbench.tools.mock.faults import FaultInjectingTool

    clock = FakeClock()
    registry = ToolRegistry([FaultInjectingTool(CalculatorTool(), failures=3, mode="transient")])
    runner = AgentRunner(registry, clock=clock)
    task = Task(key="calc", prompt="Compute 1 + 1", allowed_tools=("calculator",))
    limits = RunLimits(max_retries=3, retry_backoff_s=0.5)

    result = asyncio.run(runner.run(task, RuleBasedAgent(), limits))

    assert result.final_answer == "2"
    assert clock.sleeps == [0.5, 1.0, 2.0]
    attempts = [e.payload["attempt"] for e in result.events if e.type == "TOOL_CALL_STARTED"]
    assert attempts == [1, 2, 3, 4]


def test_identical_runs_produce_identical_results() -> None:
    def once() -> RunResult:
        runner = AgentRunner(
            default_registry(), clock=FakeClock(), id_factory=counter_ids().__next__
        )
        task = Task(
            key="pct",
            prompt="What is 17% of the quantity of record A-12?",
            allowed_tools=("lookup_record", "calculator"),
        )
        return asyncio.run(runner.run(task, RuleBasedAgent()))

    first, second = once(), once()
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_duration_and_timestamps_come_from_the_clock() -> None:
    clock = FakeClock()

    class TakesTwoSeconds:
        name, version = "slow", "1"

        async def decide(self, context: AgentContext) -> FinalAnswerDecision:
            clock.advance(2)
            return FinalAnswerDecision(answer="ok")

    runner = AgentRunner(default_registry(), clock=clock)
    result = asyncio.run(runner.run(TASK, TakesTwoSeconds()))
    assert result.duration_us == 2_000_000
    assert (result.finished_at - result.started_at).total_seconds() == 2


def usage(model: str = "example-model", inp: int = 100, out: int = 10) -> TokenUsage:
    return TokenUsage(model=model, input_tokens=inp, output_tokens=out)


def test_token_totals_when_every_decision_reports_usage() -> None:
    agent = ScriptedAgent(
        [
            ToolCallDecision(
                tool="lookup_record", arguments={"record_id": "A-12"}, usage=usage(inp=100)
            ),
            FinalAnswerDecision(answer="120", usage=usage(inp=150, out=5)),
        ]
    )
    result = run(agent)
    assert (result.input_tokens, result.output_tokens) == (250, 15)
    assert result.events[-1].payload["input_tokens"] == 250


def test_partial_usage_is_unavailable_not_a_partial_sum() -> None:
    agent = ScriptedAgent(
        [
            ToolCallDecision(tool="lookup_record", arguments={"record_id": "A-12"}, usage=usage()),
            FinalAnswerDecision(answer="120"),  # reported nothing
        ]
    )
    result = run(agent)
    assert (result.input_tokens, result.output_tokens) == (None, None)


def test_agent_without_usage_reports_none_not_zero() -> None:
    result = run(RuleBasedAgent())
    assert (result.input_tokens, result.output_tokens) == (None, None)
