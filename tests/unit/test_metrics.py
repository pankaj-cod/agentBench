"""Aggregate metric definitions: numerators, denominators, exclusions, and
unavailable values. Evaluations are built directly so each count is exact."""

import uuid
from decimal import Decimal
from typing import Any

import pytest

from agentbench.domain.benchmark import BenchmarkRef, Expectations
from agentbench.domain.evaluation import (
    CheckResult,
    Evaluation,
    EvaluationOutcome,
    FailureCategory,
    RunMetrics,
)
from agentbench.domain.models import RunLimits
from agentbench.evaluation.metrics import (
    AgentConfigRef,
    percentile_nearest_rank,
    summarize,
)

BENCH = BenchmarkRef(name="b", version="1", content_hash="0" * 64)
AGENT = AgentConfigRef(label="a", agent_name="x", agent_version="1", limits=RunLimits())


def make_eval(
    category: FailureCategory | None = None,
    execution: FailureCategory | None = None,
    evaluator_version: str = "eval-1",
    pricing_version: str | None = "none",
    **metrics: Any,
) -> Evaluation:
    base: dict[str, Any] = {
        "steps": 2,
        "duration_us": 1000,
        "tool_calls": 0,
        "tool_calls_unknown_or_not_allowed": 0,
        "tool_calls_invalid_arguments": 0,
        "tool_attempts": 0,
        "tool_attempts_failed": 0,
        "tool_selection_correct": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
    }
    base.update(metrics)
    if category is None:
        outcome, source = EvaluationOutcome.PASSED, None
    else:
        source = category.source
        outcome = EvaluationOutcome.FAILED if source.value == "AGENT" else EvaluationOutcome.INVALID
    execution = execution or (category if category and category.precedence <= 4 else None)
    checks = (
        CheckResult(
            name="execution",
            passed=execution is None,
            failure_category=execution,
            explanation="",
        ),
    )
    return Evaluation(
        run_id=uuid.uuid4(),
        case_key=f"case-{uuid.uuid4().hex[:6]}",
        evaluator_version=evaluator_version,
        pricing_version=pricing_version,
        outcome=outcome,
        failure_category=category,
        failure_source=source,
        explanation="",
        checks=checks,
        metrics=RunMetrics(**base),
        expectations=Expectations(),
    )


def summary(*evals: Evaluation) -> Any:
    return summarize(BENCH, AGENT, list(evals))


def test_empty_benchmark_reports_every_metric_as_unavailable_not_zero() -> None:
    s = summary()
    assert (s.runs, s.passed, s.failed, s.invalid) == (0, 0, 0, 0)
    for name in (
        "task_success_rate",
        "tool_selection_correctness",
        "tool_argument_validity",
        "tool_execution_error_rate",
        "timeout_rate",
        "step_limit_rate",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_mean",
        "steps_per_run_mean",
        "input_tokens_total",
        "estimated_cost_usd_total",
    ):
        metric = getattr(s, name)
        assert metric.value is None, name
        assert metric.unavailable_reason, name
    assert s.failure_distribution == {}


def test_task_success_rate_excludes_infrastructure_failures() -> None:
    s = summary(
        make_eval(),
        make_eval(),
        make_eval(FailureCategory.WRONG_ANSWER),
        make_eval(FailureCategory.TOOL_FAILURE),  # INVALID: excluded
        make_eval(FailureCategory.HARNESS_ERROR),  # INVALID: excluded
    )
    rate = s.task_success_rate
    assert (rate.numerator, rate.denominator) == (2, 3)
    assert rate.value == pytest.approx(2 / 3)
    assert (s.passed, s.failed, s.invalid) == (2, 1, 2)


def test_success_rate_is_unavailable_when_every_run_is_invalid() -> None:
    rate = summary(make_eval(FailureCategory.HARNESS_ERROR)).task_success_rate
    assert rate.value is None
    assert rate.unavailable_reason == "no valid runs"


def test_tool_selection_correctness_only_counts_cases_with_tool_expectations() -> None:
    s = summary(
        make_eval(tool_selection_correct=True),
        make_eval(FailureCategory.MISSING_REQUIRED_TOOL, tool_selection_correct=False),
        make_eval(tool_selection_correct=None),  # no tool expectations: excluded
        make_eval(FailureCategory.TOOL_FAILURE, tool_selection_correct=True),  # INVALID
    )
    rate = s.tool_selection_correctness
    assert (rate.numerator, rate.denominator) == (1, 2)


def test_argument_validity_is_per_call_and_excludes_selection_errors() -> None:
    s = summary(
        make_eval(tool_calls=3),
        make_eval(
            FailureCategory.INVALID_TOOL_CALL,
            tool_calls=4,
            tool_calls_invalid_arguments=1,
            tool_calls_unknown_or_not_allowed=1,  # not part of the denominator
        ),
    )
    rate = s.tool_argument_validity
    assert (rate.numerator, rate.denominator) == (5, 6)


def test_execution_error_rate_counts_attempts_including_invalid_runs() -> None:
    s = summary(
        make_eval(tool_attempts=4, tool_attempts_failed=1),
        make_eval(FailureCategory.TOOL_FAILURE, tool_attempts=3, tool_attempts_failed=3),
    )
    rate = s.tool_execution_error_rate
    assert (rate.numerator, rate.denominator) == (4, 7)


def test_execution_error_rate_unavailable_without_tool_executions() -> None:
    assert summary(make_eval()).tool_execution_error_rate.value is None


def test_timeout_and_step_limit_rates_use_all_runs() -> None:
    s = summary(
        make_eval(),
        make_eval(FailureCategory.TIMEOUT),
        make_eval(FailureCategory.STEP_LIMIT),
        make_eval(FailureCategory.STEP_LIMIT),
        make_eval(FailureCategory.HARNESS_ERROR),
    )
    assert (s.timeout_rate.numerator, s.timeout_rate.denominator) == (1, 5)
    assert (s.step_limit_rate.numerator, s.step_limit_rate.denominator) == (2, 5)


@pytest.mark.parametrize(
    ("values", "percentile", "expected"),
    [([4, 1, 3, 2], 50, 2), ([1, 2, 3, 4, 5], 50, 3), (list(range(1, 21)), 95, 19), ([7], 95, 7)],
)
def test_nearest_rank_percentile(values: list[int], percentile: int, expected: int) -> None:
    assert percentile_nearest_rank(values, percentile) == expected


def test_p95_needs_twenty_valid_runs() -> None:
    nineteen = [make_eval(duration_us=(i + 1) * 1000) for i in range(19)]
    s19 = summary(*nineteen)
    assert s19.latency_ms_p95.value is None
    assert s19.latency_ms_p95.unavailable_reason == "needs at least 20 valid runs, have 19"
    assert s19.latency_ms_p50.value == 10.0

    s20 = summary(*nineteen, make_eval(duration_us=20_000))
    assert s20.latency_ms_p95.value == 19.0
    assert s20.latency_ms_p95.sample_size == 20


def test_latency_and_steps_exclude_invalid_runs() -> None:
    s = summary(
        make_eval(duration_us=2000, steps=2),
        make_eval(duration_us=4000, steps=4),
        make_eval(FailureCategory.HARNESS_ERROR, duration_us=999_000, steps=0),
    )
    assert s.latency_ms_mean.value == 3.0
    assert s.steps_per_run_mean.value == 3.0
    assert s.steps_per_run_mean.sample_size == 2


def test_missing_usage_makes_totals_unavailable_not_zero() -> None:
    s = summary(make_eval(input_tokens=100, output_tokens=10), make_eval())
    assert s.input_tokens_total.value is None
    assert s.input_tokens_total.unavailable_reason == "reported by 1 of 2 runs"
    assert s.estimated_cost_usd_total.value is None


def test_usage_totals_when_every_run_reports() -> None:
    s = summary(
        make_eval(input_tokens=100, output_tokens=10, estimated_cost_usd=Decimal("0.001")),
        make_eval(
            FailureCategory.TOOL_FAILURE,
            input_tokens=50,
            output_tokens=5,
            estimated_cost_usd=Decimal("0.0005"),
        ),
    )
    assert s.input_tokens_total.value == 150
    assert s.output_tokens_total.value == 15
    assert s.estimated_cost_usd_total.value == pytest.approx(0.0015)


def test_failure_distribution_counts_primary_categories_in_precedence_order() -> None:
    s = summary(
        make_eval(FailureCategory.WRONG_ANSWER),
        make_eval(FailureCategory.WRONG_ANSWER),
        make_eval(FailureCategory.STEP_LIMIT),
        make_eval(),
    )
    assert list(s.failure_distribution.items()) == [
        (FailureCategory.STEP_LIMIT, 1),
        (FailureCategory.WRONG_ANSWER, 2),
    ]


def test_mixing_evaluator_or_pricing_versions_is_rejected() -> None:
    with pytest.raises(ValueError, match="evaluator versions"):
        summary(make_eval(), make_eval(evaluator_version="eval-2"))
    with pytest.raises(ValueError, match="price table"):
        summary(make_eval(), make_eval(pricing_version="prices-2"))
