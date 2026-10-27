"""Plain-text rendering of benchmark summaries and comparisons."""

from agentbench.evaluation.comparison import BenchmarkComparison, MetricComparison
from agentbench.evaluation.metrics import BenchmarkSummary, Rate, Statistic

RATE_NAMES = (
    "task_success_rate",
    "tool_selection_correctness",
    "tool_argument_validity",
    "tool_execution_error_rate",
    "timeout_rate",
    "step_limit_rate",
)
VALUE_NAMES = (
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_mean",
    "steps_per_run_mean",
    "input_tokens_total",
    "output_tokens_total",
    "estimated_cost_usd_total",
)


def _rate(rate: Rate) -> str:
    if rate.value is None:
        return f"unavailable ({rate.unavailable_reason})"
    return f"{rate.value * 100:6.1f}%  ({rate.numerator}/{rate.denominator})"


def _stat(stat: Statistic) -> str:
    if stat.value is None:
        return f"unavailable ({stat.unavailable_reason})"
    return f"{stat.value:.3f}  (n={stat.sample_size})"


def render_summary(s: BenchmarkSummary) -> str:
    lines = [
        f"benchmark  {s.benchmark.name} v{s.benchmark.version} ({s.benchmark.content_hash[:12]})",
        f"agent      {s.agent.label}: {s.agent.agent_name}/{s.agent.agent_version}, "
        f"max_steps={s.agent.limits.max_steps}",
        f"evaluator  {s.evaluator_version}, prices {s.pricing_version}",
        f"runs       {s.runs}: {s.passed} passed, {s.failed} failed, {s.invalid} invalid",
        "",
    ]
    width = max(len(n) for n in RATE_NAMES + VALUE_NAMES)
    lines += [f"{n:<{width}}  {_rate(getattr(s, n))}" for n in RATE_NAMES]
    lines += [f"{n:<{width}}  {_stat(getattr(s, n))}" for n in VALUE_NAMES]
    lines += [
        "",
        "failures   "
        + (", ".join(f"{c}={n}" for c, n in s.failure_distribution.items()) or "none"),
        "",
    ]
    lines.append(f"{'case':<26} {'outcome':<8} {'ms':>8} {'steps':>5}  explanation")
    for c in s.cases:
        lines.append(
            f"{c.case_key:<26} {c.outcome:<8} {c.duration_ms:>8.3f} {c.steps:>5}  {c.explanation}"
        )
    return "\n".join(lines) + "\n"


def _change(m: MetricComparison) -> str:
    if m.absolute_change is None:
        return f"n/a ({m.note})" if m.note else "n/a"
    if m.kind == "rate":
        return f"{m.percentage_point_change:+.1f} pp"
    relative = (
        f" ({m.relative_change_percent:+.1f}%)" if m.relative_change_percent is not None else ""
    )
    return f"{m.absolute_change:+.3f} {m.unit}{relative}"


def _value(m: MetricComparison, value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if m.kind == "rate" else f"{value:.3f}"


def render_comparison(c: BenchmarkComparison) -> str:
    lines = [
        f"benchmark  {c.benchmark.name} v{c.benchmark.version} ({c.benchmark.content_hash[:12]}), "
        f"evaluator {c.evaluator_version}",
        f"baseline   {c.baseline.label} (max_steps={c.baseline.limits.max_steps})",
        f"candidate  {c.candidate.label} (max_steps={c.candidate.limits.max_steps})",
        "",
        f"{'metric':<26} {'baseline':>10} {'candidate':>10}  change",
    ]
    for m in c.metrics:
        lines.append(
            f"{m.metric:<26} {_value(m, m.baseline):>10} {_value(m, m.candidate):>10}  {_change(m)}"
        )
    lines += ["", f"{'failure category':<26} {'baseline':>10} {'candidate':>10}  change"]
    for f in c.failure_distribution:
        lines.append(f"{f.category:<26} {f.baseline:>10} {f.candidate:>10}  {f.change:+d}")
    lines += ["", f"{'case':<26} {'baseline':<22} {'candidate':<22} change"]
    for case in c.cases:
        base = f"{case.baseline_outcome} {case.baseline_category or ''}".strip()
        cand = f"{case.candidate_outcome} {case.candidate_category or ''}".strip()
        lines.append(f"{case.case_key:<26} {base:<22} {cand:<22} {case.change}")
    lines += ["", f"fixed: {list(c.fixed)}", f"broken: {list(c.broken)}"]
    return "\n".join(lines) + "\n"
