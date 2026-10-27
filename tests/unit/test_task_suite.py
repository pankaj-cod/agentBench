"""End-to-end runner scenarios: each task runs through the real AgentRunner, real
ToolRegistry, real mock tools, and real TraceRecorder, with a FakeClock.

Each case states the expected termination reason, final answer, and the exact sequence
of trace event types, so any change in runner behaviour shows up as a precise diff.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from agentbench.agents.base import Agent, AgentContext
from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.agents.scripted import ScriptedAgent
from agentbench.domain.enums import AgentRunStatus, TerminationReason
from agentbench.domain.models import (
    FinalAnswerDecision,
    RunLimits,
    RunResult,
    Task,
    ToolCallDecision,
)
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import DEFAULT_RECORDS, RecordStore, default_registry
from agentbench.tools.mock.calculator import CalculatorTool
from agentbench.tools.mock.faults import FaultInjectingTool, FaultMode
from agentbench.tools.mock.records import LookupRecordTool, SearchRecordsTool
from agentbench.tools.registry import ToolRegistry
from agentbench.tracing.clock import FakeClock

ALL_TOOLS = ("search_records", "lookup_record", "calculator")

# Event type abbreviations keep the expected sequences readable.
RS, AD, TS, OK, TF, VE, FA, RF = (
    "RUN_STARTED",
    "AGENT_DECISION",
    "TOOL_CALL_STARTED",
    "TOOL_CALL_SUCCEEDED",
    "TOOL_CALL_FAILED",
    "VALIDATION_ERROR",
    "FINAL_ANSWER",
    "RUN_FINISHED",
)


def faulty_lookup_registry(mode: FaultMode, failures: int) -> ToolRegistry:
    store = RecordStore(DEFAULT_RECORDS)
    lookup = FaultInjectingTool(LookupRecordTool(store), failures=failures, mode=mode)
    return ToolRegistry([SearchRecordsTool(store), lookup, CalculatorTool()])


class SlowAgent:
    """Wraps an agent and advances the fake clock on every decision."""

    def __init__(self, inner: Agent, clock: FakeClock, seconds: float) -> None:
        self.inner, self.clock, self.seconds = inner, clock, seconds
        self.name, self.version = inner.name, inner.version

    async def decide(self, context: AgentContext) -> ToolCallDecision | FinalAnswerDecision:
        self.clock.advance(self.seconds)
        return await self.inner.decide(context)


def call(tool: str, arguments: object) -> ToolCallDecision:
    return ToolCallDecision(tool=tool, arguments=arguments)


def answer(text: str) -> FinalAnswerDecision:
    return FinalAnswerDecision(answer=text)


@dataclass
class Case:
    key: str
    prompt: str
    reason: TerminationReason
    events: list[str]
    answer: str | None = None
    agent: Callable[[FakeClock], Agent] = lambda clock: RuleBasedAgent()
    registry: Callable[[], ToolRegistry] = default_registry
    allowed: tuple[str, ...] = ALL_TOOLS
    limits: RunLimits = field(default_factory=RunLimits)
    sleeps: list[float] = field(default_factory=list)


CASES = [
    # --- successful lookups ---
    Case(
        "lookup-quantity",
        "What is the quantity of record A-12?",
        TerminationReason.COMPLETED,
        [RS, AD, TS, OK, AD, FA, RF],
        answer="120",
    ),
    Case(
        "lookup-price",
        "what is the price of record b-07",
        TerminationReason.COMPLETED,
        [RS, AD, TS, OK, AD, FA, RF],
        answer="6.5",
    ),
    Case(
        "search-by-name",
        "Find records matching 'cable'",
        TerminationReason.COMPLETED,
        [RS, AD, TS, OK, AD, FA, RF],
        answer="B-07, B-09",
    ),
    # --- multi-step tool use ---
    Case(
        "percent-of-quantity",
        "What is 17% of the quantity of record A-12?",
        TerminationReason.COMPLETED,
        [RS, AD, TS, OK, AD, TS, OK, AD, FA, RF],
        answer="20.4",
    ),
    Case(
        "total-stock-value",
        "What is the total value of records matching 'cable'?",
        TerminationReason.COMPLETED,
        [RS, AD, TS, OK, AD, TS, OK, AD, FA, RF],
        answer="468",  # 45 * 6.5 + 18 * 9.75
    ),
    Case(
        "compute-expression",
        "Compute (12 + 8) * 3",
        TerminationReason.COMPLETED,
        [RS, AD, TS, OK, AD, FA, RF],
        answer="60",
    ),
    # --- tool errors are reported to the agent, never replaced by a value ---
    Case(
        "missing-record",
        "What is the quantity of record Z-99?",
        TerminationReason.COMPLETED,
        [RS, AD, TS, TF, AD, FA, RF],
        answer="unable to answer: ToolExecutionError: record not found: Z-99",
    ),
    Case(
        "permanent-tool-failure-not-retried",
        "What is the quantity of record A-12?",
        TerminationReason.COMPLETED,
        [RS, AD, TS, TF, AD, FA, RF],
        answer="unable to answer: ToolExecutionError: injected permanent failure 1/1",
        registry=lambda: faulty_lookup_registry("permanent", failures=1),
    ),
    # --- incorrect arguments ---
    Case(
        "code-injection-rejected-by-calculator",
        "Compute __import__('os').system('ls')",
        TerminationReason.COMPLETED,
        [RS, AD, VE, AD, FA, RF],
        answer="unable to answer: ToolArgumentsError: invalid arguments for calculator",
    ),
    Case(
        "wrong-argument-name-then-recovery",
        "Look up A-12 (scripted)",
        TerminationReason.COMPLETED,
        [RS, AD, VE, AD, TS, OK, AD, FA, RF],
        answer="120",
        agent=lambda clock: ScriptedAgent(
            [
                call("lookup_record", {"id": "A-12"}),
                call("lookup_record", {"record_id": "A-12"}),
                answer("120"),
            ]
        ),
    ),
    Case(
        "malformed-arguments",
        "Look up A-12 with a bare string (scripted)",
        TerminationReason.COMPLETED,
        [RS, AD, VE, AD, FA, RF],
        answer="gave up",
        agent=lambda clock: ScriptedAgent([call("lookup_record", "A-12"), answer("gave up")]),
    ),
    # --- unknown or forbidden tools ---
    Case(
        "unknown-tool",
        "Delete a record (scripted)",
        TerminationReason.COMPLETED,
        [RS, AD, VE, AD, FA, RF],
        answer="cannot delete",
        agent=lambda clock: ScriptedAgent(
            [call("delete_record", {"record_id": "A-12"}), answer("cannot delete")]
        ),
    ),
    Case(
        "tool-not-allowed-for-task",
        "What is the quantity of record A-12?",
        TerminationReason.COMPLETED,
        [RS, AD, VE, AD, FA, RF],
        answer=(
            "unable to answer: ToolNotAllowedError: tool lookup_record is not allowed for this task"
        ),
        allowed=("calculator",),
    ),
    # --- tool failures and retries ---
    Case(
        "transient-failure-recovered-by-retry",
        "What is the quantity of record A-12?",
        TerminationReason.COMPLETED,
        [RS, AD, TS, TF, TS, OK, AD, FA, RF],
        answer="120",
        registry=lambda: faulty_lookup_registry("transient", failures=1),
        sleeps=[0.1],
    ),
    Case(
        "transient-failure-exhausts-retries",
        "What is the quantity of record A-12?",
        TerminationReason.TOOL_FAILURE_EXHAUSTED,
        [RS, AD, TS, TF, TS, TF, TS, TF, RF],
        registry=lambda: faulty_lookup_registry("transient", failures=99),
        sleeps=[0.1, 0.2],
    ),
    Case(
        "hanging-tool-times-out-every-attempt",
        "What is the quantity of record A-12?",
        TerminationReason.TOOL_FAILURE_EXHAUSTED,
        [RS, AD, TS, TF, TS, TF, RF],
        registry=lambda: faulty_lookup_registry("hang", failures=99),
        limits=RunLimits(tool_timeout_s=0.01, max_retries=1),
        sleeps=[0.1],
    ),
    # --- limits ---
    Case(
        "step-limit",
        "Search forever (scripted)",
        TerminationReason.MAX_STEPS_EXCEEDED,
        [RS, AD, TS, OK, AD, TS, OK, AD, TS, OK, RF],
        agent=lambda clock: ScriptedAgent([call("search_records", {"query": "steel"})] * 10),
        limits=RunLimits(max_steps=3),
    ),
    Case(
        "run-timeout",
        "Find records matching 'steel'",
        TerminationReason.RUN_TIMEOUT,
        # Step 1 decides at t=40s and searches. Step 2's decision arrives at t=80s, past
        # the 60s deadline: it is recorded, but its tool call is never started.
        [RS, AD, TS, OK, AD, RF],
        agent=lambda clock: SlowAgent(
            ScriptedAgent([call("search_records", {"query": "steel"})] * 10), clock, seconds=40
        ),
        limits=RunLimits(run_timeout_s=60),
    ),
    # --- agent failures ---
    Case(
        "unsupported-task",
        "Write a poem about bolts",
        TerminationReason.AGENT_ERROR,
        [RS, RF],
    ),
]


def run_case(case: Case) -> tuple[RunResult, FakeClock]:
    clock = FakeClock()
    runner = AgentRunner(case.registry(), clock=clock)
    task = Task(key=case.key, prompt=case.prompt, allowed_tools=case.allowed)
    result = asyncio.run(runner.run(task, case.agent(clock), case.limits))
    return result, clock


def test_suite_has_at_least_ten_distinct_tasks() -> None:
    assert len(CASES) >= 10
    assert len({c.key for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.key)
def test_task(case: Case) -> None:
    result, clock = run_case(case)

    assert result.termination_reason is case.reason, result.error_message
    assert [e.type.value for e in result.events] == case.events
    assert result.final_answer == case.answer
    assert clock.sleeps == case.sleeps
    if case.reason is TerminationReason.COMPLETED:
        assert result.error_message is None
    else:
        assert result.status is AgentRunStatus.FAILED
        assert result.error_message  # a failure always says why


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.key)
def test_summary_counts_match_the_trace(case: Case) -> None:
    result, _ = run_case(case)
    decisions = [e for e in result.events if e.type == "AGENT_DECISION"]
    tool_decisions = [d for d in decisions if d.payload["kind"] == "tool_call"]
    finished = result.events[-1]

    assert result.tool_call_count == len(tool_decisions)
    assert result.step_count >= len(decisions)
    assert finished.type == "RUN_FINISHED"
    assert finished.payload["termination_reason"] == result.termination_reason
    assert finished.payload["step_count"] == result.step_count


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.key)
def test_no_success_event_without_a_real_tool_success(case: Case) -> None:
    """Every TOOL_CALL_SUCCEEDED follows a TOOL_CALL_STARTED for the same call and
    attempt; failed calls never produce an output."""
    result, _ = run_case(case)
    started = set()
    for event in result.events:
        key = (event.payload.get("call_id"), event.payload.get("attempt"))
        if event.type == "TOOL_CALL_STARTED":
            started.add(key)
        elif event.type == "TOOL_CALL_SUCCEEDED":
            assert key in started
        elif event.type == "TOOL_CALL_FAILED":
            assert key in started
            assert "output" not in event.payload
