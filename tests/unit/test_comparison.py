from typing import Any

import pytest

from agentbench.domain.benchmark import BenchmarkRef
from agentbench.domain.evaluation import EvaluationOutcome, FailureCategory
from agentbench.domain.models import RunLimits
from agentbench.evaluation.comparison import CaseChange, ComparisonError, compare
from agentbench.evaluation.metrics import AgentConfigRef, BenchmarkSummary, summarize

from .test_metrics import BENCH, make_eval


def summary_of(
    label: str, outcomes: dict[str, FailureCategory | None], **kw: Any
) -> BenchmarkSummary:
    evals = []
    for key, category in outcomes.items():
        evaluation = make_eval(category, **kw.get(key, {}))
        evals.append(evaluation.model_copy(update={"case_key": key}))
    agent = AgentConfigRef(label=label, agent_name="x", agent_version="1", limits=RunLimits())
    return summarize(kw.get("bench", BENCH), agent, evals)


BASE = {
    "a": None,
    "b": FailureCategory.WRONG_ANSWER,
    "c": None,
    "d": FailureCategory.STEP_LIMIT,
}
CAND = {
    "a": None,
    "b": None,  # fixed
    "c": FailureCategory.MISSING_REQUIRED_TOOL,  # broken
    "d": FailureCategory.TOOL_FAILURE,  # INVALID: not comparable
}


def metric(comparison: Any, name: str) -> Any:
    return next(m for m in comparison.metrics if m.metric == name)


def test_rates_report_absolute_and_percentage_point_change_but_no_relative_change() -> None:
    result = compare(summary_of("base", BASE), summary_of("cand", CAND))
    success = metric(result, "task_success_rate")
    # baseline 2/4 = 0.5; candidate 2/3 (d is INVALID and excluded)
    assert success.baseline == 0.5
    assert success.candidate == pytest.approx(0.666667)
    assert success.absolute_change == pytest.approx(0.166667)
    assert success.percentage_point_change == pytest.approx(16.666667)
    assert success.relative_change_percent is None


def test_values_report_absolute_and_relative_change() -> None:
    base = summary_of(
        "base", {"a": None, "b": None}, a={"duration_us": 10_000}, b={"duration_us": 10_000}
    )
    cand = summary_of(
        "cand", {"a": None, "b": None}, a={"duration_us": 15_000}, b={"duration_us": 15_000}
    )
    latency = metric(compare(base, cand), "latency_ms_mean")
    assert (latency.baseline, latency.candidate) == (10.0, 15.0)
    assert latency.absolute_change == 5.0
    assert latency.relative_change_percent == 50.0
    assert latency.percentage_point_change is None


def test_unavailable_metrics_have_no_change_and_say_why() -> None:
    result = compare(summary_of("base", BASE), summary_of("cand", CAND))
    cost = metric(result, "estimated_cost_usd_total")
    assert cost.absolute_change is None
    assert "reported by 0 of 4 runs" in cost.note
    p95 = metric(result, "latency_ms_p95")
    assert p95.absolute_change is None
    assert "needs at least 20" in p95.note


def test_zero_baseline_has_no_relative_change() -> None:
    base = summary_of("base", {"a": None}, a={"steps": 0})
    cand = summary_of("cand", {"a": None}, a={"steps": 2})
    steps = metric(compare(base, cand), "steps_per_run_mean")
    assert steps.absolute_change == 2.0
    assert steps.relative_change_percent is None
    assert steps.note == "relative change undefined: baseline is 0"


def test_per_case_changes() -> None:
    result = compare(summary_of("base", BASE), summary_of("cand", CAND))
    changes = {c.case_key: c.change for c in result.cases}
    assert changes == {
        "a": CaseChange.STILL_PASSING,
        "b": CaseChange.FIXED,
        "c": CaseChange.BROKEN,
        "d": CaseChange.NOT_COMPARABLE,
    }
    assert (result.fixed, result.broken) == (("b",), ("c",))
    d = next(c for c in result.cases if c.case_key == "d")
    assert d.candidate_outcome is EvaluationOutcome.INVALID


def test_failure_distribution_changes_include_categories_from_either_side() -> None:
    result = compare(summary_of("base", BASE), summary_of("cand", CAND))
    rows = {f.category: (f.baseline, f.candidate, f.change) for f in result.failure_distribution}
    assert rows == {
        FailureCategory.TOOL_FAILURE: (0, 1, 1),
        FailureCategory.STEP_LIMIT: (1, 0, -1),
        FailureCategory.MISSING_REQUIRED_TOOL: (0, 1, 1),
        FailureCategory.WRONG_ANSWER: (1, 0, -1),
    }


def test_different_benchmark_content_cannot_be_compared() -> None:
    other = BenchmarkRef(name="b", version="1", content_hash="1" * 64)
    with pytest.raises(ComparisonError, match="different benchmarks"):
        compare(summary_of("base", BASE), summary_of("cand", BASE, bench=other))


def test_different_evaluator_versions_cannot_be_compared() -> None:
    newer = {k: {"evaluator_version": "eval-2"} for k in BASE}
    with pytest.raises(ComparisonError, match="evaluator versions differ"):
        compare(summary_of("base", BASE), summary_of("cand", BASE, **newer))


def test_different_case_sets_cannot_be_compared() -> None:
    with pytest.raises(ComparisonError, match="case sets differ"):
        compare(summary_of("base", BASE), summary_of("cand", {"a": None}))


def test_empty_summaries_cannot_be_compared() -> None:
    with pytest.raises(ComparisonError, match="empty"):
        compare(summary_of("base", {}), summary_of("cand", {}))
