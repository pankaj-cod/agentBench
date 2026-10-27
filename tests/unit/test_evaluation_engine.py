"""EvaluationEngine tests. Every run here is a real trace produced by AgentRunner, so
the evaluator is tested against exactly what the runner records."""

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.agents.scripted import ScriptedAgent
from agentbench.domain.benchmark import (
    ExactAnswer,
    Expectations,
    NumericAnswer,
    OneOfAnswer,
    RegexAnswer,
    TestCase,
    ToolExpectation,
)
from agentbench.domain.evaluation import (
    Evaluation,
    EvaluationOutcome,
    FailureCategory,
    FailureSource,
)
from agentbench.domain.models import (
    FinalAnswerDecision,
    RunLimits,
    RunResult,
    TokenUsage,
    ToolCallDecision,
)
from agentbench.evaluation.engine import EVALUATOR_VERSION, EvaluationEngine
from agentbench.evaluation.pricing import PriceTable
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import DEFAULT_RECORDS, RecordStore, default_registry
from agentbench.tools.mock.calculator import CalculatorTool
from agentbench.tools.mock.faults import FaultInjectingTool, FaultMode
from agentbench.tools.mock.records import LookupRecordTool, SearchRecordsTool
from agentbench.tools.registry import ToolRegistry
from agentbench.tracing.clock import FakeClock

ALL_TOOLS = ("search_records", "lookup_record", "calculator")
PCT_PROMPT = "What is 17% of the quantity of record A-12?"


def case(prompt: str = PCT_PROMPT, **expectations: Any) -> TestCase:
    return TestCase(
        key="case",
        prompt=prompt,
        allowed_tools=ALL_TOOLS,
        expectations=Expectations(**expectations),
    )


def call(tool: str, usage: TokenUsage | None = None, **arguments: Any) -> ToolCallDecision:
    return ToolCallDecision(tool=tool, arguments=arguments, usage=usage)


def answer(text: str, usage: TokenUsage | None = None) -> FinalAnswerDecision:
    return FinalAnswerDecision(answer=text, usage=usage)


def execute(
    test_case: TestCase,
    agent: Any = None,
    registry: ToolRegistry | None = None,
    limits: RunLimits | None = None,
) -> RunResult:
    runner = AgentRunner(registry or default_registry(), clock=FakeClock())
    return asyncio.run(runner.run(test_case.to_task(), agent or RuleBasedAgent(), limits))


def evaluate(test_case: TestCase, run: RunResult, **engine_kwargs: Any) -> Evaluation:
    return EvaluationEngine(**engine_kwargs).evaluate(test_case, run)


def score(test_case: TestCase, agent: Any = None, **kwargs: Any) -> Evaluation:
    return evaluate(test_case, execute(test_case, agent, **kwargs))


def flaky_lookup(mode: FaultMode) -> ToolRegistry:
    store = RecordStore(DEFAULT_RECORDS)
    lookup = FaultInjectingTool(LookupRecordTool(store), failures=99, mode=mode)
    return ToolRegistry([SearchRecordsTool(store), lookup, CalculatorTool()])


FULL_PCT = {
    "final_answer": NumericAnswer(value=20.4, tolerance=0.01),
    "required_tools": (
        ToolExpectation(name="lookup_record", args_subset={"record_id": "A-12"}),
        ToolExpectation(name="calculator"),
    ),
    "expected_records": ("A-12",),
    "max_tool_calls": 3,
}


# --- correct behaviour -----------------------------------------------------------


def test_correct_multi_step_run_passes_every_check() -> None:
    result = score(case(**FULL_PCT))
    assert result.outcome is EvaluationOutcome.PASSED
    assert result.failure_category is None and result.failure_source is None
    assert [c.name for c in result.checks] == [
        "execution",
        "final_answer",
        "required_tool:lookup_record",
        "required_tool:calculator",
        "valid_tool_calls",
        "max_tool_calls",
        "retrieved_records",
    ]
    assert result.evaluator_version == EVALUATOR_VERSION
    assert result.expectations == case(**FULL_PCT).expectations


# --- final answers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "passed"),
    [("20.4", True), ("20.405", True), (" 20.40 ", True), ("20.42", False), ("17", False)],
)
def test_numeric_answer_tolerance(given: str, passed: bool) -> None:
    test_case = case(final_answer=NumericAnswer(value=20.4, tolerance=0.01))
    result = score(test_case, ScriptedAgent([answer(given)]))
    assert (result.outcome is EvaluationOutcome.PASSED) is passed
    if not passed:
        assert result.failure_category is FailureCategory.WRONG_ANSWER


@pytest.mark.parametrize("given", ["about 20.4", "20.4 units", "nan", "inf", ""])
def test_numeric_answer_must_be_exactly_a_finite_number(given: str) -> None:
    test_case = case(final_answer=NumericAnswer(value=20.4, tolerance=1))
    result = score(test_case, ScriptedAgent([answer(given)]))
    assert result.failure_category is FailureCategory.WRONG_ANSWER


@pytest.mark.parametrize(
    ("expectation", "given", "passed"),
    [
        (ExactAnswer(value="Steel bolt M8"), "steel BOLT m8", True),
        (ExactAnswer(value="Steel bolt M8", case_sensitive=True), "steel bolt m8", False),
        (RegexAnswer(pattern=r"B-07,\s*B-09"), "B-07, B-09", True),
        (RegexAnswer(pattern=r"B-07"), "B-07, B-09", False),  # must match the whole answer
        (OneOfAnswer(values=("none", "no records")), " None ", True),
        (OneOfAnswer(values=("none",)), "B-07", False),
    ],
)
def test_other_answer_types(expectation: Any, given: str, passed: bool) -> None:
    result = score(case(final_answer=expectation), ScriptedAgent([answer(given)]))
    assert (result.outcome is EvaluationOutcome.PASSED) is passed


def test_incorrect_answer_explanation_shows_expected_and_actual() -> None:
    test_case = case(final_answer=NumericAnswer(value=20.4, tolerance=0.01))
    result = score(test_case, ScriptedAgent([answer("17")]))
    assert result.explanation == "WRONG_ANSWER: expected 20.4 ± 0.01, got '17'"


# --- tool use ---------------------------------------------------------------------


def test_missing_required_tool_call() -> None:
    # Right answer, but guessed without looking anything up.
    result = score(case(**FULL_PCT), ScriptedAgent([answer("20.4")]))
    assert result.outcome is EvaluationOutcome.FAILED
    assert result.failure_category is FailureCategory.MISSING_REQUIRED_TOOL
    assert result.failure_source is FailureSource.AGENT
    assert result.metrics.tool_selection_correct is False


def test_required_tool_with_wrong_arguments() -> None:
    agent = ScriptedAgent(
        [
            call("lookup_record", record_id="A-15"),  # valid call, wrong record
            call("calculator", expression="120 * 17 / 100"),
            answer("20.4"),
        ]
    )
    result = score(case(**FULL_PCT), agent)
    assert result.failure_category is FailureCategory.WRONG_TOOL_ARGUMENTS
    assert result.metrics.tool_selection_correct is True  # right tool, wrong arguments


def test_invalid_arguments_fail_even_when_the_answer_is_right() -> None:
    agent = ScriptedAgent(
        [
            call("lookup_record", id="A-12"),  # invalid: wrong argument name
            call("lookup_record", record_id="A-12"),
            call("calculator", expression="120 * 17 / 100"),
            answer("20.4"),
        ]
    )
    result = score(case(**{**FULL_PCT, "max_tool_calls": None}), agent)
    assert result.failure_category is FailureCategory.INVALID_TOOL_CALL
    assert result.metrics.tool_calls_invalid_arguments == 1
    assert result.metrics.tool_calls_unknown_or_not_allowed == 0


def test_invalid_calls_can_be_allowed_explicitly() -> None:
    agent = ScriptedAgent([call("lookup_record", id="A-12"), answer("x")])
    result = score(case(allow_invalid_tool_calls=True), agent)
    assert result.outcome is EvaluationOutcome.PASSED


def test_unknown_tool_counts_as_a_selection_error_not_an_argument_error() -> None:
    agent = ScriptedAgent([call("delete_record", record_id="A-12"), answer("x")])
    result = score(case(forbidden_tools=("calculator",)), agent)
    assert result.failure_category is FailureCategory.INVALID_TOOL_CALL
    assert result.metrics.tool_calls_unknown_or_not_allowed == 1
    assert result.metrics.tool_calls_invalid_arguments == 0
    assert result.metrics.tool_selection_correct is False


def test_forbidden_tool_used() -> None:
    test_case = case("Compute (12 + 8) * 3", forbidden_tools=("search_records",))
    agent = ScriptedAgent(
        [call("search_records", query="x"), call("calculator", expression="60"), answer("60")]
    )
    result = score(test_case, agent)
    assert result.failure_category is FailureCategory.FORBIDDEN_TOOL_USED


def test_too_many_tool_calls() -> None:
    agent = ScriptedAgent([call("search_records", query="steel")] * 3 + [answer("A-12")])
    result = score(case(max_tool_calls=2), agent)
    assert result.failure_category is FailureCategory.TOOL_CALL_LIMIT_EXCEEDED


def test_missing_expected_retrieved_records() -> None:
    test_case = case(
        "Find records matching 'cable'",
        final_answer=RegexAnswer(pattern=".*"),
        expected_records=("B-07", "B-09"),
    )
    agent = ScriptedAgent([call("lookup_record", record_id="B-07"), answer("B-07")])
    result = score(test_case, agent)
    assert result.failure_category is FailureCategory.MISSING_RETRIEVED_RECORDS
    assert "B-09" in result.explanation


def test_failed_lookup_does_not_count_as_retrieved() -> None:
    test_case = case(expected_records=("Z-99",))
    result = score(test_case, ScriptedAgent([call("lookup_record", record_id="Z-99"), answer("x")]))
    assert result.failure_category is FailureCategory.MISSING_RETRIEVED_RECORDS


# --- execution failures: agent vs infrastructure ------------------------------------


def test_step_limit_is_an_agent_failure() -> None:
    result = score(case(**FULL_PCT), limits=RunLimits(max_steps=1))
    assert result.outcome is EvaluationOutcome.FAILED
    assert result.failure_category is FailureCategory.STEP_LIMIT
    # The missing answer and calculator call are listed, but the step limit is primary.
    assert "also failed" in result.explanation


def test_agent_exception_is_an_agent_failure() -> None:
    result = score(case("Write a poem", final_answer=ExactAnswer(value="x")))
    assert result.failure_category is FailureCategory.AGENT_ERROR
    assert result.failure_source is FailureSource.AGENT


def test_run_timeout_is_an_agent_failure() -> None:
    clock = FakeClock()

    class Slow:
        name, version = "slow", "1"

        async def decide(self, context: Any) -> FinalAnswerDecision:
            clock.advance(10)
            return await ScriptedAgent([call("search_records", query="a")] * 5).decide(context)

    runner = AgentRunner(default_registry(), clock=clock)
    test_case = case(final_answer=ExactAnswer(value="x"))
    run = asyncio.run(runner.run(test_case.to_task(), Slow(), RunLimits(run_timeout_s=15)))
    result = evaluate(test_case, run)
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.outcome is EvaluationOutcome.FAILED


def test_exhausted_tool_retries_are_an_infrastructure_failure() -> None:
    result = score(case(**FULL_PCT), registry=flaky_lookup("transient"))
    assert result.outcome is EvaluationOutcome.INVALID
    assert result.failure_category is FailureCategory.TOOL_FAILURE
    assert result.failure_source is FailureSource.INFRASTRUCTURE
    assert result.metrics.tool_attempts == 3
    assert result.metrics.tool_attempts_failed == 3


def test_harness_error_is_an_infrastructure_failure() -> None:
    class BrokenRegistry(ToolRegistry):
        def schemas(self, names: Any) -> list[dict[str, Any]]:
            raise RuntimeError("bug in AgentBench")

    registry = BrokenRegistry(default_registry()._tools.values())
    result = score(case(**FULL_PCT), registry=registry)
    assert result.outcome is EvaluationOutcome.INVALID
    assert result.failure_category is FailureCategory.HARNESS_ERROR


def test_permanent_tool_error_caused_by_the_agent_is_an_agent_failure() -> None:
    # The agent asks for a record that does not exist: its own mistake.
    test_case = case("What is the quantity of record Z-99?", final_answer=NumericAnswer(value=120))
    result = score(test_case)
    assert result.outcome is EvaluationOutcome.FAILED
    # The lookup failed permanently, so the agent answered "unable to answer: ...".
    assert result.failure_category is FailureCategory.WRONG_ANSWER
    assert "unable to answer: ToolExecutionError" in result.explanation
    assert result.metrics.tool_attempts_failed == 1


# --- usage and cost ---------------------------------------------------------------


def test_missing_usage_is_unavailable_not_zero() -> None:
    result = score(case(**FULL_PCT))
    assert result.metrics.input_tokens is None
    assert result.metrics.output_tokens is None
    assert result.metrics.estimated_cost_usd is None


PRICES = PriceTable("test-prices-1", {"example-model": (Decimal("2.00"), Decimal("10.00"))})
USAGE = TokenUsage(model="example-model", input_tokens=1000, output_tokens=100)


def test_cost_is_computed_when_every_decision_reports_a_priced_model() -> None:
    agent = ScriptedAgent([call("search_records", usage=USAGE, query="x"), answer("x", USAGE)])
    test_case = case()
    result = evaluate(test_case, execute(test_case, agent), prices=PRICES)
    # 2 decisions x (1000 * $2 + 100 * $10) / 1M = 2 x $0.003
    assert result.metrics.estimated_cost_usd == Decimal("0.006000")
    assert (result.metrics.input_tokens, result.metrics.output_tokens) == (2000, 200)
    assert result.pricing_version == "test-prices-1"


def test_cost_is_unavailable_for_an_unpriced_model_or_partial_usage() -> None:
    other = TokenUsage(model="unknown-model", input_tokens=1, output_tokens=1)
    unpriced = ScriptedAgent([answer("x", other)])
    partial = ScriptedAgent([call("search_records", usage=USAGE, query="x"), answer("x")])
    test_case = case()
    for agent in (unpriced, partial):
        result = evaluate(test_case, execute(test_case, agent), prices=PRICES)
        assert result.metrics.estimated_cost_usd is None


# --- determinism ------------------------------------------------------------------


def test_repeated_evaluation_of_the_same_trace_is_identical() -> None:
    test_case = case(**FULL_PCT)
    run = execute(test_case)
    first, second = evaluate(test_case, run), evaluate(test_case, run)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_evaluation_rejects_a_run_of_a_different_case() -> None:
    run = execute(case())
    other = TestCase(key="other", prompt="x", expectations=Expectations())
    with pytest.raises(ValueError, match="not case"):
        evaluate(other, run)
