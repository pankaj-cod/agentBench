"""Four different things that are easy to confuse:

TRACE                Re-read a stored run and score it again. Nothing is executed:
                     no model call, no tool call. Fully deterministic.
RECORDED_RESPONSES   Run the task again, executing the tools again, but answer every
                     model request with the response recorded in the original run.
                     Deterministic as long as the tools are; any divergence in what
                     would have been sent to the model is reported.
LIVE_SAME_CONFIG     Run the task again with the same versioned configuration and a
                     live model. NOT deterministic: the model may answer differently.
                     Same config_hash means same configuration, not same behaviour.
NEW_CONFIG           Run the task with a different model, prompt, or settings: a new
                     config_hash. This is an experiment, not a replay.
"""

from enum import StrEnum

from agentbench.agents.model_agent import ModelAgent
from agentbench.domain.agent_config import ModelAgentConfig
from agentbench.domain.benchmark import TestCase
from agentbench.domain.enums import ExecutionMode, TraceEventType
from agentbench.domain.errors import AgentBenchError
from agentbench.domain.evaluation import Evaluation
from agentbench.domain.models import DomainModel, RunResult
from agentbench.evaluation.engine import EvaluationEngine
from agentbench.experiments.recorded import RecordedResponseAdapter
from agentbench.llm.base import ModelAdapter
from agentbench.runner.runner import AgentRunner
from agentbench.tracing.clock import Clock


class ReplayMode(StrEnum):
    TRACE = "TRACE"
    RECORDED_RESPONSES = "RECORDED_RESPONSES"
    LIVE_SAME_CONFIG = "LIVE_SAME_CONFIG"
    NEW_CONFIG = "NEW_CONFIG"


class ReplayError(AgentBenchError):
    pass


class RunDifference(DomainModel):
    same_termination: bool
    same_final_answer: bool
    same_tool_calls: bool  # same tools with the same arguments, in the same order
    same_tool_outputs: bool
    model_request_divergence_steps: tuple[int, ...] = ()  # RECORDED_RESPONSES only


class ReplayResult(DomainModel):
    mode: ReplayMode
    source_run_id: str
    new_run: RunResult | None  # None for TRACE: nothing was executed
    evaluation: Evaluation
    difference: RunDifference | None
    deterministic: bool  # only ever True when no live model call was made
    note: str


def config_from_run(run: RunResult) -> ModelAgentConfig:
    """The versioned configuration recorded in RUN_STARTED, verified against its hash."""
    started = run.events[0]
    if started.type is not TraceEventType.RUN_STARTED:
        raise ReplayError("run has no RUN_STARTED event")
    recorded = started.payload["agent"].get("config")
    if not recorded:
        raise ReplayError(f"run {run.run_id} was not made by a configurable model agent")
    config = ModelAgentConfig.model_validate(
        {k: v for k, v in recorded.items() if k != "config_hash"}
    )
    if config.config_hash != recorded["config_hash"] or config.config_hash != run.agent_config_hash:
        raise ReplayError("recorded configuration does not match its hash")
    return config


def _tool_calls(run: RunResult) -> list[tuple[str, object]]:
    return [
        (e.payload["tool"], e.payload["arguments"])
        for e in run.events
        if e.type is TraceEventType.TOOL_CALL_STARTED
    ]


def _tool_outputs(run: RunResult) -> list[object]:
    return [e.payload["output"] for e in run.events if e.type is TraceEventType.TOOL_CALL_SUCCEEDED]


def compare_runs(original: RunResult, new: RunResult, divergence: list[int]) -> RunDifference:
    return RunDifference(
        same_termination=original.termination_reason is new.termination_reason,
        same_final_answer=original.final_answer == new.final_answer,
        same_tool_calls=_tool_calls(original) == _tool_calls(new),
        same_tool_outputs=_tool_outputs(original) == _tool_outputs(new),
        model_request_divergence_steps=tuple(divergence),
    )


class ReplayService:
    def __init__(self, runner: AgentRunner, engine: EvaluationEngine, clock: Clock | None = None):
        self._runner = runner
        self._engine = engine
        self._clock = clock

    def replay_trace(self, run: RunResult, case: TestCase) -> ReplayResult:
        return ReplayResult(
            mode=ReplayMode.TRACE,
            source_run_id=str(run.run_id),
            new_run=None,
            evaluation=self._engine.evaluate(case, run),
            difference=None,
            deterministic=True,
            note="stored trace re-scored; nothing was executed",
        )

    async def reexecute_with_recorded_responses(
        self, run: RunResult, case: TestCase
    ) -> ReplayResult:
        config = config_from_run(run)
        adapter = RecordedResponseAdapter(run, provider=config.provider)
        new = await self._execute(case, config, adapter, run, ExecutionMode.RECORDED_RESPONSES)
        difference = compare_runs(run, new, adapter.divergent_steps)
        diverged = bool(adapter.divergent_steps) or not difference.same_tool_outputs
        return ReplayResult(
            mode=ReplayMode.RECORDED_RESPONSES,
            source_run_id=str(run.run_id),
            new_run=new,
            evaluation=self._engine.evaluate(case, new),
            difference=difference,
            deterministic=not diverged,
            note=(
                "tools re-executed with recorded model responses"
                + (
                    f"; conversation diverged at steps {adapter.divergent_steps}"
                    if diverged
                    else ""
                )
            ),
        )

    async def reexecute_live(
        self,
        run: RunResult,
        case: TestCase,
        adapter: ModelAdapter,
        new_config: ModelAgentConfig | None = None,
    ) -> ReplayResult:
        original = config_from_run(run)
        config = new_config or original
        same = config.config_hash == original.config_hash
        new = await self._execute(case, config, adapter, run, ExecutionMode.LIVE)
        return ReplayResult(
            mode=ReplayMode.LIVE_SAME_CONFIG if same else ReplayMode.NEW_CONFIG,
            source_run_id=str(run.run_id),
            new_run=new,
            evaluation=self._engine.evaluate(case, new),
            difference=compare_runs(run, new, []),
            deterministic=False,
            note=(
                "live re-execution with the same configuration; the model may answer "
                "differently, so differences are expected and are not by themselves a bug"
                if same
                else f"new configuration {config.version} (was {original.version})"
            ),
        )

    async def _execute(
        self,
        case: TestCase,
        config: ModelAgentConfig,
        adapter: ModelAdapter,
        source: RunResult,
        mode: ExecutionMode,
    ) -> RunResult:
        if case.key != source.task.key:
            raise ReplayError(f"case {case.key!r} is not the task of run {source.run_id}")
        agent = ModelAgent(config, adapter, clock=self._clock)
        return await self._runner.run(
            case.to_task(), agent, source.limits, execution_mode=mode, source_run_id=source.run_id
        )
