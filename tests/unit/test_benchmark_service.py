import asyncio
import json
from pathlib import Path

import pytest

from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.benchmarks.service import (
    AgentConfig,
    BenchmarkLoadError,
    BenchmarkRunner,
    load_benchmark,
)
from agentbench.domain.evaluation import EvaluationOutcome, FailureCategory
from agentbench.domain.models import RunLimits
from agentbench.evaluation.comparison import CaseChange, compare
from agentbench.evaluation.engine import EvaluationEngine
from agentbench.evaluation.metrics import BenchmarkSummary
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import default_registry
from agentbench.tracing.clock import FakeClock

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "benchmarks" / "inventory-v1.json"
THREE_STEP_CASES = {
    "percent-of-quantity-a12",
    "percent-of-price-b09",
    "total-quantity-steel",
    "total-value-cables",
}
UNSUPPORTED = {"paraphrased-lookup", "percent-of-total"}


def run(max_steps: int, label: str) -> BenchmarkSummary:
    runner = BenchmarkRunner(AgentRunner(default_registry(), clock=FakeClock()), EvaluationEngine())
    config = AgentConfig(label, RuleBasedAgent, RunLimits(max_steps=max_steps))
    return asyncio.run(runner.run(load_benchmark(EXAMPLE), config))


def test_example_benchmark_loads() -> None:
    benchmark = load_benchmark(EXAMPLE)
    assert len(benchmark.cases) == 14
    assert len(benchmark.content_hash) == 64


def test_rule_based_agent_on_example_benchmark() -> None:
    summary = run(10, "default")
    assert (summary.runs, summary.passed, summary.failed, summary.invalid) == (14, 12, 2, 0)
    failed = {c.case_key: c.failure_category for c in summary.cases if c.failure_category}
    assert failed == {key: FailureCategory.AGENT_ERROR for key in UNSUPPORTED}
    assert summary.task_success_rate.value == pytest.approx(12 / 14)
    # The agent reports no usage, so these are unavailable, not zero.
    assert summary.input_tokens_total.value is None
    assert summary.estimated_cost_usd_total.value is None
    # 14 runs < 20, so p95 is unavailable.
    assert summary.latency_ms_p95.value is None


def test_two_step_budget_breaks_exactly_the_three_step_cases() -> None:
    comparison = compare(run(10, "default"), run(2, "two-steps"))
    assert set(comparison.broken) == THREE_STEP_CASES
    assert comparison.fixed == ()
    for case in comparison.cases:
        if case.case_key in THREE_STEP_CASES:
            assert case.candidate_category is FailureCategory.STEP_LIMIT
        elif case.case_key in UNSUPPORTED:
            assert case.change is CaseChange.STILL_FAILING
        else:
            assert case.change is CaseChange.STILL_PASSING
    success = next(m for m in comparison.metrics if m.metric == "task_success_rate")
    assert success.percentage_point_change == pytest.approx(-4 / 14 * 100, abs=1e-4)
    step_limit = next(m for m in comparison.metrics if m.metric == "step_limit_rate")
    assert (step_limit.baseline, step_limit.candidate) == (0.0, pytest.approx(4 / 14, abs=1e-6))


def test_every_case_outcome_is_reported_with_an_explanation() -> None:
    summary = run(10, "default")
    assert [c.case_key for c in summary.cases] == [c.key for c in load_benchmark(EXAMPLE).cases]
    for case in summary.cases:
        assert case.explanation
        if case.outcome is EvaluationOutcome.PASSED:
            assert case.explanation.startswith("all ")


def test_invalid_benchmark_file_gives_a_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "b", "version": "1", "cases": []}))
    with pytest.raises(BenchmarkLoadError, match="at least 1"):
        load_benchmark(bad)
    with pytest.raises(BenchmarkLoadError, match="cannot read"):
        load_benchmark(tmp_path / "missing.json")
