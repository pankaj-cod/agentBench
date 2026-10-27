"""Compare two agent configurations on the same benchmark.

Rates report an absolute change (candidate - baseline, as a fraction) and the same
change in percentage points. Other values (latency, steps, tokens, cost) report an
absolute change in their own unit and a relative change in percent. Relative change
is never reported for rates: going from 50% to 75% is +25 percentage points, and
calling it "+50%" invites misreading.
"""

from enum import StrEnum
from typing import Literal

from agentbench.domain.benchmark import BenchmarkRef
from agentbench.domain.errors import AgentBenchError
from agentbench.domain.evaluation import EvaluationOutcome, FailureCategory
from agentbench.domain.models import DomainModel
from agentbench.evaluation.metrics import AgentConfigRef, BenchmarkSummary, Rate, Statistic


class ComparisonError(AgentBenchError):
    pass


class CaseChange(StrEnum):
    FIXED = "FIXED"  # FAILED -> PASSED
    BROKEN = "BROKEN"  # PASSED -> FAILED
    STILL_PASSING = "STILL_PASSING"
    STILL_FAILING = "STILL_FAILING"
    NOT_COMPARABLE = "NOT_COMPARABLE"  # INVALID on either side


class MetricComparison(DomainModel):
    metric: str
    kind: Literal["rate", "value"]
    unit: str
    baseline: float | None
    candidate: float | None
    absolute_change: float | None
    percentage_point_change: float | None  # rates only
    relative_change_percent: float | None  # values only, and only if baseline != 0
    note: str | None = None


class FailureCount(DomainModel):
    category: FailureCategory
    baseline: int
    candidate: int
    change: int


class CaseComparison(DomainModel):
    case_key: str
    baseline_outcome: EvaluationOutcome
    candidate_outcome: EvaluationOutcome
    baseline_category: FailureCategory | None
    candidate_category: FailureCategory | None
    change: CaseChange


class BenchmarkComparison(DomainModel):
    benchmark: BenchmarkRef
    evaluator_version: str
    baseline: AgentConfigRef
    candidate: AgentConfigRef
    metrics: tuple[MetricComparison, ...]
    failure_distribution: tuple[FailureCount, ...]
    cases: tuple[CaseComparison, ...]
    fixed: tuple[str, ...]
    broken: tuple[str, ...]


RATES = (
    "task_success_rate",
    "tool_selection_correctness",
    "tool_argument_validity",
    "tool_execution_error_rate",
    "timeout_rate",
    "step_limit_rate",
)
VALUES = (
    ("latency_ms_p50", "ms"),
    ("latency_ms_p95", "ms"),
    ("latency_ms_mean", "ms"),
    ("steps_per_run_mean", "steps"),
    ("input_tokens_total", "tokens"),
    ("output_tokens_total", "tokens"),
    ("estimated_cost_usd_total", "USD"),
)


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _unavailable_note(base: Rate | Statistic, cand: Rate | Statistic) -> str | None:
    reasons = []
    if base.value is None:
        reasons.append(f"baseline: {base.unavailable_reason}")
    if cand.value is None:
        reasons.append(f"candidate: {cand.unavailable_reason}")
    return "; ".join(reasons) or None


def _compare_rate(name: str, base: Rate, cand: Rate) -> MetricComparison:
    change = None if base.value is None or cand.value is None else cand.value - base.value
    return MetricComparison(
        metric=name,
        kind="rate",
        unit="fraction",
        baseline=_r(base.value),
        candidate=_r(cand.value),
        absolute_change=_r(change),
        percentage_point_change=_r(None if change is None else change * 100),
        relative_change_percent=None,
        note=_unavailable_note(base, cand),
    )


def _compare_value(name: str, unit: str, base: Statistic, cand: Statistic) -> MetricComparison:
    b, c = base.value, cand.value
    change = None if b is None or c is None else c - b
    relative = None
    note = _unavailable_note(base, cand)
    if change is not None and b is not None:
        if b == 0:
            note = "relative change undefined: baseline is 0"
        else:
            relative = change / abs(b) * 100
    return MetricComparison(
        metric=name,
        kind="value",
        unit=unit,
        baseline=_r(base.value),
        candidate=_r(cand.value),
        absolute_change=_r(change),
        percentage_point_change=None,
        relative_change_percent=_r(relative),
        note=note,
    )


def _case_change(base: EvaluationOutcome, cand: EvaluationOutcome) -> CaseChange:
    if EvaluationOutcome.INVALID in (base, cand):
        return CaseChange.NOT_COMPARABLE
    if base is cand:
        return (
            CaseChange.STILL_PASSING
            if base is EvaluationOutcome.PASSED
            else CaseChange.STILL_FAILING
        )
    return CaseChange.FIXED if cand is EvaluationOutcome.PASSED else CaseChange.BROKEN


def compare(baseline: BenchmarkSummary, candidate: BenchmarkSummary) -> BenchmarkComparison:
    if baseline.benchmark != candidate.benchmark:
        raise ComparisonError(
            "summaries are for different benchmarks or benchmark contents: "
            f"{baseline.benchmark.name}@{baseline.benchmark.content_hash[:12]} vs "
            f"{candidate.benchmark.name}@{candidate.benchmark.content_hash[:12]}"
        )
    if baseline.evaluator_version != candidate.evaluator_version:
        raise ComparisonError(
            f"evaluator versions differ: {baseline.evaluator_version} vs "
            f"{candidate.evaluator_version}; re-score both with the same evaluator"
        )
    if baseline.evaluator_version is None:
        raise ComparisonError("cannot compare empty summaries")
    base_cases = {c.case_key: c for c in baseline.cases}
    cand_cases = {c.case_key: c for c in candidate.cases}
    if base_cases.keys() != cand_cases.keys():
        missing = sorted(base_cases.keys() ^ cand_cases.keys())
        raise ComparisonError(f"case sets differ: {missing}")

    cases = tuple(
        CaseComparison(
            case_key=key,
            baseline_outcome=base_cases[key].outcome,
            candidate_outcome=cand_cases[key].outcome,
            baseline_category=base_cases[key].failure_category,
            candidate_category=cand_cases[key].failure_category,
            change=_case_change(base_cases[key].outcome, cand_cases[key].outcome),
        )
        for key in base_cases
    )
    categories = sorted(
        baseline.failure_distribution.keys() | candidate.failure_distribution.keys(),
        key=lambda c: c.precedence,
    )
    return BenchmarkComparison(
        benchmark=baseline.benchmark,
        evaluator_version=baseline.evaluator_version,
        baseline=baseline.agent,
        candidate=candidate.agent,
        metrics=tuple(
            [_compare_rate(n, getattr(baseline, n), getattr(candidate, n)) for n in RATES]
            + [
                _compare_value(n, unit, getattr(baseline, n), getattr(candidate, n))
                for n, unit in VALUES
            ]
        ),
        failure_distribution=tuple(
            FailureCount(
                category=c,
                baseline=baseline.failure_distribution.get(c, 0),
                candidate=candidate.failure_distribution.get(c, 0),
                change=candidate.failure_distribution.get(c, 0)
                - baseline.failure_distribution.get(c, 0),
            )
            for c in categories
        ),
        cases=cases,
        fixed=tuple(c.case_key for c in cases if c.change is CaseChange.FIXED),
        broken=tuple(c.case_key for c in cases if c.change is CaseChange.BROKEN),
    )
