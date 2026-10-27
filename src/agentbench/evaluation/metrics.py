"""Aggregate metrics over the evaluations of one agent configuration on one benchmark.

Every metric states its numerator, denominator and exclusions. A metric that cannot be
measured is reported with `value=None` and a reason, never as 0. "Valid runs" means
runs whose outcome is PASSED or FAILED; INVALID runs (infrastructure failures) say
nothing about the agent and are excluded wherever noted.
"""

import math
from collections import Counter
from collections.abc import Callable, Sequence
from decimal import Decimal

from pydantic import Field

from agentbench.domain.benchmark import BenchmarkRef
from agentbench.domain.evaluation import (
    Evaluation,
    EvaluationOutcome,
    FailureCategory,
    FailureSource,
)
from agentbench.domain.models import DomainModel, RunLimits

P95_MIN_SAMPLES = 20


class Rate(DomainModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None  # numerator / denominator, None when the denominator is 0
    unavailable_reason: str | None = None

    @classmethod
    def of(cls, numerator: int, denominator: int, empty_reason: str) -> "Rate":
        if denominator == 0:
            return cls(numerator=0, denominator=0, value=None, unavailable_reason=empty_reason)
        return cls(numerator=numerator, denominator=denominator, value=numerator / denominator)


class Statistic(DomainModel):
    value: float | None
    sample_size: int = Field(ge=0)
    unavailable_reason: str | None = None

    @classmethod
    def unavailable(cls, sample_size: int, reason: str) -> "Statistic":
        return cls(value=None, sample_size=sample_size, unavailable_reason=reason)


class AgentConfigRef(DomainModel):
    label: str
    agent_name: str
    agent_version: str
    limits: RunLimits


class CaseResult(DomainModel):
    case_key: str
    run_id: str
    outcome: EvaluationOutcome
    failure_category: FailureCategory | None
    failure_source: FailureSource | None
    explanation: str
    duration_ms: float
    steps: int


class BenchmarkSummary(DomainModel):
    benchmark: BenchmarkRef
    agent: AgentConfigRef
    evaluator_version: str | None  # None only for an empty summary
    pricing_version: str | None
    runs: int
    passed: int
    failed: int
    invalid: int
    task_success_rate: Rate
    tool_selection_correctness: Rate
    tool_argument_validity: Rate
    tool_execution_error_rate: Rate
    timeout_rate: Rate
    step_limit_rate: Rate
    latency_ms_p50: Statistic
    latency_ms_p95: Statistic
    latency_ms_mean: Statistic
    steps_per_run_mean: Statistic
    input_tokens_total: Statistic
    output_tokens_total: Statistic
    estimated_cost_usd_total: Statistic
    failure_distribution: dict[FailureCategory, int]
    cases: tuple[CaseResult, ...]


# --- metric definitions -------------------------------------------------------------


def _valid(evals: Sequence[Evaluation]) -> list[Evaluation]:
    return [e for e in evals if e.outcome is not EvaluationOutcome.INVALID]


def task_success_rate(evals: Sequence[Evaluation]) -> Rate:
    """PASSED runs / valid runs. Excludes INVALID runs."""
    valid = _valid(evals)
    passed = sum(e.outcome is EvaluationOutcome.PASSED for e in valid)
    return Rate.of(passed, len(valid), "no valid runs")


def tool_selection_correctness(evals: Sequence[Evaluation]) -> Rate:
    """Valid runs whose tool selection was correct / valid runs whose test case has
    tool expectations. Correct means: every required tool was requested, no forbidden
    tool was requested, and no unknown or not-allowed tool was requested. Excludes
    INVALID runs and cases with no required or forbidden tools."""
    judged = [e for e in _valid(evals) if e.metrics.tool_selection_correct is not None]
    correct = sum(bool(e.metrics.tool_selection_correct) for e in judged)
    return Rate.of(correct, len(judged), "no valid runs with tool expectations")


def tool_argument_validity(evals: Sequence[Evaluation]) -> Rate:
    """Tool calls whose arguments passed validation / tool calls naming a registered,
    allowed tool. Counted per call across valid runs. Excludes INVALID runs and calls to
    unknown or not-allowed tools (those are selection errors, not argument errors)."""
    valid = _valid(evals)
    eligible = sum(
        e.metrics.tool_calls - e.metrics.tool_calls_unknown_or_not_allowed for e in valid
    )
    invalid_args = sum(e.metrics.tool_calls_invalid_arguments for e in valid)
    return Rate.of(eligible - invalid_args, eligible, "no tool calls to registered, allowed tools")


def tool_execution_error_rate(evals: Sequence[Evaluation]) -> Rate:
    """Failed tool executions / tool executions started, counted per attempt (retries
    included) across ALL runs, including INVALID ones: infrastructure failures are
    exactly what this rate is meant to show. Calls rejected by validation never execute
    and are not counted."""
    started = sum(e.metrics.tool_attempts for e in evals)
    failed = sum(e.metrics.tool_attempts_failed for e in evals)
    return Rate.of(failed, started, "no tool executions")


def _execution_category(evaluation: Evaluation) -> FailureCategory | None:
    return next(c.failure_category for c in evaluation.checks if c.name == "execution")


def termination_rate(evals: Sequence[Evaluation], category: FailureCategory) -> Rate:
    """Runs that ended with the given termination (TIMEOUT or STEP_LIMIT) / all runs,
    including INVALID runs."""
    hits = sum(_execution_category(e) is category for e in evals)
    return Rate.of(hits, len(evals), "no runs")


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    """The smallest value such that at least `percentile`% of values are <= it."""
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def latency_ms(evals: Sequence[Evaluation], percentile: float, min_samples: int) -> Statistic:
    """Nearest-rank percentile of end-to-end run duration in ms, over valid runs.
    Excludes INVALID runs. Unavailable below `min_samples` (1 for p50, 20 for p95: with
    fewer than 20 samples the nearest-rank p95 is just the maximum)."""
    samples = [e.metrics.duration_us / 1000 for e in _valid(evals)]
    if len(samples) < min_samples:
        return Statistic.unavailable(
            len(samples), f"needs at least {min_samples} valid runs, have {len(samples)}"
        )
    return Statistic(value=percentile_nearest_rank(samples, percentile), sample_size=len(samples))


def _mean(evals: Sequence[Evaluation], value: Callable[[Evaluation], float]) -> Statistic:
    valid = _valid(evals)
    if not valid:
        return Statistic.unavailable(0, "no valid runs")
    return Statistic(value=sum(value(e) for e in valid) / len(valid), sample_size=len(valid))


def latency_ms_mean(evals: Sequence[Evaluation]) -> Statistic:
    """Mean end-to-end duration in ms over valid runs. Excludes INVALID runs."""
    return _mean(evals, lambda e: e.metrics.duration_us / 1000)


def steps_per_run_mean(evals: Sequence[Evaluation]) -> Statistic:
    """Mean agent decisions per run over valid runs. Excludes INVALID runs."""
    return _mean(evals, lambda e: e.metrics.steps)


def usage_total(
    evals: Sequence[Evaluation], value: Callable[[Evaluation], int | Decimal | None]
) -> Statistic:
    """Sum over ALL runs (spend is spend, even on INVALID runs). Unavailable unless every
    run reported the value; a partial sum would understate usage."""
    values = [value(e) for e in evals]
    reported = [v for v in values if v is not None]
    if not values:
        return Statistic.unavailable(0, "no runs")
    if len(reported) < len(values):
        return Statistic.unavailable(
            len(reported), f"reported by {len(reported)} of {len(values)} runs"
        )
    return Statistic(value=float(sum(reported)), sample_size=len(reported))


# --- summary ------------------------------------------------------------------------


def summarize(
    benchmark: BenchmarkRef, agent: AgentConfigRef, evals: Sequence[Evaluation]
) -> BenchmarkSummary:
    versions = {e.evaluator_version for e in evals}
    if len(versions) > 1:
        raise ValueError(f"cannot mix evaluator versions in one summary: {sorted(versions)}")
    pricing = {e.pricing_version for e in evals}
    if len(pricing) > 1:
        raise ValueError(f"cannot mix price table versions in one summary: {sorted(pricing)}")
    outcomes = Counter(e.outcome for e in evals)
    return BenchmarkSummary(
        benchmark=benchmark,
        agent=agent,
        evaluator_version=next(iter(versions), None),
        pricing_version=next(iter(pricing), None),
        runs=len(evals),
        passed=outcomes[EvaluationOutcome.PASSED],
        failed=outcomes[EvaluationOutcome.FAILED],
        invalid=outcomes[EvaluationOutcome.INVALID],
        task_success_rate=task_success_rate(evals),
        tool_selection_correctness=tool_selection_correctness(evals),
        tool_argument_validity=tool_argument_validity(evals),
        tool_execution_error_rate=tool_execution_error_rate(evals),
        timeout_rate=termination_rate(evals, FailureCategory.TIMEOUT),
        step_limit_rate=termination_rate(evals, FailureCategory.STEP_LIMIT),
        latency_ms_p50=latency_ms(evals, 50, min_samples=1),
        latency_ms_p95=latency_ms(evals, 95, min_samples=P95_MIN_SAMPLES),
        latency_ms_mean=latency_ms_mean(evals),
        steps_per_run_mean=steps_per_run_mean(evals),
        input_tokens_total=usage_total(evals, lambda e: e.metrics.input_tokens),
        output_tokens_total=usage_total(evals, lambda e: e.metrics.output_tokens),
        estimated_cost_usd_total=usage_total(evals, lambda e: e.metrics.estimated_cost_usd),
        failure_distribution=dict(
            sorted(
                Counter(e.failure_category for e in evals if e.failure_category).items(),
                key=lambda item: item[0].precedence,
            )
        ),
        cases=tuple(
            CaseResult(
                case_key=e.case_key,
                run_id=str(e.run_id),
                outcome=e.outcome,
                failure_category=e.failure_category,
                failure_source=e.failure_source,
                explanation=e.explanation,
                duration_ms=e.metrics.duration_us / 1000,
                steps=e.metrics.steps,
            )
            for e in evals
        ),
    )
