"""Command-line interface.

    agentbench run "What is the quantity of record A-12?" [--save] [--json]
    agentbench show <run-id> [--json]
    agentbench bench <benchmark.json> --label NAME [--max-steps N] [--save] [--output FILE]
    agentbench compare <baseline-summary.json> <candidate-summary.json> [--output FILE]

`run`/`show` exit with 0 if the run COMPLETED and 1 otherwise. `bench`/`compare` exit
with 0 on success and 1 if the input cannot be used. Usage errors exit with 2.
"""

import argparse
import asyncio
import json
import sys
import uuid
from functools import partial
from pathlib import Path
from typing import Any, TextIO

from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.benchmarks.service import (
    AgentConfig,
    BenchmarkLoadError,
    BenchmarkRunner,
    load_benchmark,
)
from agentbench.domain.enums import AgentRunStatus, TraceEventType
from agentbench.domain.models import RunLimits, RunResult, Task, TraceEvent
from agentbench.evaluation.comparison import ComparisonError, compare
from agentbench.evaluation.engine import EvaluationEngine
from agentbench.evaluation.metrics import BenchmarkSummary
from agentbench.persistence.db import create_db_engine, create_session_factory, database_url
from agentbench.persistence.uow import UnitOfWork
from agentbench.reports import render_comparison, render_summary
from agentbench.runner.runner import AgentRunner
from agentbench.service import TaskRunService
from agentbench.tools.mock import default_registry

ALL_TOOLS = ("search_records", "lookup_record", "calculator")


def main(argv: list[str] | None = None, out: TextIO = sys.stdout) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bench":
        return _bench(args, out)
    if args.command == "compare":
        return _compare(args, out)
    if args.command == "run":
        result = asyncio.run(_run(args))
    else:
        result = _load(args.run_id)
        if result is None:
            print(f"run not found: {args.run_id}", file=sys.stderr)
            return 1
    if args.json:
        out.write(result.model_dump_json(indent=2) + "\n")
    else:
        out.write(render(result))
    return 0 if result.status is AgentRunStatus.COMPLETED else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentbench")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one task with the rule-based agent")
    run.add_argument("prompt")
    run.add_argument("--key", default="adhoc", help="task key (a slug)")
    run.add_argument("--tools", nargs="+", choices=ALL_TOOLS, default=list(ALL_TOOLS))
    run.add_argument("--max-steps", type=int, default=10)
    run.add_argument("--save", action="store_true", help="save to AGENTBENCH_DATABASE_URL")
    run.add_argument("--json", action="store_true", help="print the full result as JSON")

    show = commands.add_parser("show", help="show a saved run")
    show.add_argument("run_id", type=uuid.UUID)
    show.add_argument("--json", action="store_true")

    bench = commands.add_parser("bench", help="evaluate the rule-based agent on a benchmark")
    bench.add_argument("benchmark", type=Path)
    bench.add_argument("--label", required=True, help="name of this agent configuration")
    bench.add_argument("--max-steps", type=int, default=10)
    bench.add_argument("--save", action="store_true", help="save runs and evaluations")
    bench.add_argument("--output", type=Path, help="write the summary as JSON")

    compare_cmd = commands.add_parser("compare", help="compare two benchmark summaries")
    compare_cmd.add_argument("baseline", type=Path)
    compare_cmd.add_argument("candidate", type=Path)
    compare_cmd.add_argument("--output", type=Path, help="write the comparison as JSON")
    return parser


def _bench(args: argparse.Namespace, out: TextIO) -> int:
    try:
        benchmark = load_benchmark(args.benchmark)
    except BenchmarkLoadError as exc:
        print(exc, file=sys.stderr)
        return 1
    uow_factory = None
    if args.save:
        uow_factory = partial(UnitOfWork, create_session_factory(create_db_engine(database_url())))
    runner = BenchmarkRunner(AgentRunner(default_registry()), EvaluationEngine(), uow_factory)
    config = AgentConfig(args.label, RuleBasedAgent, RunLimits(max_steps=args.max_steps))
    summary = asyncio.run(runner.run(benchmark, config))
    if args.output:
        args.output.write_text(summary.model_dump_json(indent=2) + "\n")
    out.write(render_summary(summary))
    return 0


def _compare(args: argparse.Namespace, out: TextIO) -> int:
    try:
        baseline = BenchmarkSummary.model_validate_json(args.baseline.read_text())
        candidate = BenchmarkSummary.model_validate_json(args.candidate.read_text())
        comparison = compare(baseline, candidate)
    except (OSError, ValueError, ComparisonError) as exc:
        print(f"cannot compare: {exc}", file=sys.stderr)
        return 1
    if args.output:
        args.output.write_text(comparison.model_dump_json(indent=2) + "\n")
    out.write(render_comparison(comparison))
    return 0


async def _run(args: argparse.Namespace) -> RunResult:
    task = Task(key=args.key, prompt=args.prompt, allowed_tools=tuple(args.tools))
    limits = RunLimits(max_steps=args.max_steps)
    runner = AgentRunner(default_registry())
    if not args.save:
        return await runner.run(task, RuleBasedAgent(), limits)
    sessions = create_session_factory(create_db_engine(database_url()))
    service = TaskRunService(runner, lambda: UnitOfWork(sessions))
    return await service.run_and_save(task, RuleBasedAgent(), limits)


def _load(run_id: uuid.UUID) -> RunResult | None:
    sessions = create_session_factory(create_db_engine(database_url()))
    with UnitOfWork(sessions) as uow:
        return uow.agent_runs.get(run_id)


def render(result: RunResult) -> str:
    lines = [
        f"run      {result.run_id}",
        f"task     {result.task.key}: {result.task.prompt}",
        f"agent    {result.agent_name}/{result.agent_version}",
        f"status   {result.status} ({result.termination_reason})",
        f"answer   {result.final_answer if result.final_answer is not None else '-'}",
    ]
    if result.error_message:
        lines.append(f"error    {result.error_message}")
    lines.append(
        f"steps    {result.step_count}, tool calls {result.tool_call_count}, "
        f"{result.duration_us / 1000:.3f} ms"
    )
    lines.append("")
    lines.append(f"{'seq':>3}  {'step':>4}  {'+ms':>9}  {'event':<20}  detail")
    for event in result.events:
        lines.append(
            f"{event.seq:>3}  {event.step:>4}  {event.elapsed_us / 1000:>9.3f}  "
            f"{event.type:<20}  {_detail(event)}"
        )
    return "\n".join(lines) + "\n"


def _detail(event: TraceEvent) -> str:
    p: dict[str, Any] = event.payload
    match event.type:
        case TraceEventType.RUN_STARTED:
            return f"tools={p['task']['allowed_tools']} max_steps={p['limits']['max_steps']}"
        case TraceEventType.AGENT_DECISION if p["kind"] == "tool_call":
            return f"call {p['tool']} {_compact(p['arguments'])}"
        case TraceEventType.AGENT_DECISION:
            return f"final answer {p['answer']!r}"
        case TraceEventType.TOOL_CALL_STARTED:
            return f"{p['call_id']} attempt {p['attempt']}"
        case TraceEventType.TOOL_CALL_SUCCEEDED:
            return f"{p['call_id']} -> {_compact(p['output'])}"
        case TraceEventType.TOOL_CALL_FAILED:
            return f"{p['call_id']} attempt {p['attempt']}: {_error(p)}"
        case TraceEventType.VALIDATION_ERROR:
            return f"{p['call_id']} {p['tool']}: {_error(p)}"
        case TraceEventType.FINAL_ANSWER:
            return repr(p["answer"])
        case TraceEventType.RUN_FINISHED:
            return str(p["termination_reason"])
    return _compact(p)


def _error(payload: dict[str, Any]) -> str:
    return f"{payload['error']['type']}: {payload['error']['message']}"


def _compact(value: Any, limit: int = 90) -> str:
    text = json.dumps(value, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
