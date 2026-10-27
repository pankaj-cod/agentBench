"""Replay modes and experiment comparison, all with scripted models (no network)."""

import asyncio
import re
from typing import Any

import pytest

from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.benchmarks.service import BenchmarkRunner
from agentbench.domain.agent_config import ModelAgentConfig, PromptTemplate
from agentbench.domain.benchmark import (
    Benchmark,
    Expectations,
    NumericAnswer,
    TestCase,
    ToolExpectation,
)
from agentbench.domain.enums import ExecutionMode, TerminationReason
from agentbench.domain.evaluation import EvaluationOutcome
from agentbench.domain.models import RunLimits, RunResult
from agentbench.evaluation.engine import EvaluationEngine
from agentbench.experiments.compare import (
    LIVE_MODEL_NOTE,
    ConfiguredAgent,
    ExperimentService,
)
from agentbench.experiments.recorded import RecordedResponseAdapter, RecordingExhaustedError
from agentbench.experiments.replay import ReplayError, ReplayMode, ReplayService
from agentbench.llm.base import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ToolResultsMessage,
    UserMessage,
)
from agentbench.llm.scripted import ScriptedModelAdapter
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import DEFAULT_RECORDS, Record, RecordStore, default_registry
from agentbench.tools.registry import ToolRegistry
from agentbench.tracing.clock import FakeClock

PROMPT = PromptTemplate(id="inventory", version="1", system="Use the tools.")
CONFIG = ModelAgentConfig(label="v1", provider="scripted", model="fake-model-1", prompt=PROMPT)
CASE = TestCase(
    key="pct",
    prompt="What is 17% of the quantity of record A-12?",
    allowed_tools=("lookup_record", "calculator"),
    expectations=Expectations(
        final_answer=NumericAnswer(value=20.4, tolerance=0.01),
        required_tools=(ToolExpectation(name="lookup_record", args_subset={"record_id": "A-12"}),),
    ),
)


def tool_use(name: str, arguments: dict[str, Any], call_id: str) -> ModelResponse:
    call = ModelToolCall(id=call_id, name=name, arguments=arguments)
    return ModelResponse(text=None, tool_calls=(call,), stop_reason="tool_use", usage=None)


def final(text: str) -> ModelResponse:
    return ModelResponse(text=text, tool_calls=(), stop_reason="end_turn", usage=None)


SCRIPT = [
    tool_use("lookup_record", {"record_id": "A-12"}, "t1"),
    tool_use("calculator", {"expression": "120 * 17 / 100"}, "t2"),
    final("20.4"),
]


def service(registry: ToolRegistry | None = None) -> ReplayService:
    clock = FakeClock()
    runner = AgentRunner(registry or default_registry(), clock=clock)
    return ReplayService(runner, EvaluationEngine(), clock=clock)


def original_run(script: list[Any] | None = None) -> RunResult:
    from agentbench.agents.model_agent import ModelAgent

    agent = ModelAgent(CONFIG, ScriptedModelAdapter(script or SCRIPT), clock=FakeClock())
    runner = AgentRunner(default_registry(), clock=FakeClock())
    return asyncio.run(runner.run(CASE.to_task(), agent))


# --- TRACE: re-score only -------------------------------------------------------------


def test_trace_replay_executes_nothing_and_matches_the_original_score() -> None:
    run = original_run()
    first = service().replay_trace(run, CASE)
    second = service().replay_trace(run, CASE)
    assert first.mode is ReplayMode.TRACE
    assert first.new_run is None and first.deterministic
    assert first.evaluation == second.evaluation == EvaluationEngine().evaluate(CASE, run)


# --- RECORDED_RESPONSES: tools re-run, model replayed ---------------------------------


def test_recorded_reexecution_reproduces_the_run_without_calling_a_model() -> None:
    run = original_run()
    result = asyncio.run(service().reexecute_with_recorded_responses(run, CASE))
    assert result.mode is ReplayMode.RECORDED_RESPONSES
    assert result.deterministic is True
    assert result.difference is not None
    assert result.difference.same_final_answer and result.difference.same_tool_calls
    assert result.difference.same_tool_outputs
    assert result.difference.model_request_divergence_steps == ()
    new = result.new_run
    assert new is not None and new.run_id != run.run_id
    assert new.execution_mode is ExecutionMode.RECORDED_RESPONSES
    assert new.source_run_id == run.run_id
    assert new.agent_config_hash == run.agent_config_hash
    assert result.evaluation.outcome is EvaluationOutcome.PASSED


def test_recorded_reexecution_reports_divergence_when_tool_data_changed() -> None:
    changed = [
        r.model_copy(update={"quantity": 200}) if r.id == "A-12" else r for r in DEFAULT_RECORDS
    ]
    registry = default_registry(RecordStore(changed))
    run = original_run()
    result = asyncio.run(service(registry).reexecute_with_recorded_responses(run, CASE))
    assert result.deterministic is False
    assert result.difference is not None
    assert result.difference.same_tool_outputs is False
    # Step 2's request now contains quantity 200, so it differs from the recorded one,
    # and step 3's request carries that change forward.
    assert result.difference.model_request_divergence_steps == (2, 3)
    # The recorded model still said 20.4: replayed responses can be stale.
    assert result.new_run is not None and result.new_run.final_answer == "20.4"
    assert "diverged" in result.note


def test_recording_exhausted_is_an_infrastructure_error() -> None:
    run = original_run()
    adapter = RecordedResponseAdapter(run, provider="scripted")
    request = ModelRequest(
        model="m", system="s", messages=(), tools=(), decoding=CONFIG.decoding, timeout_s=1
    )
    for _ in range(3):
        asyncio.run(adapter.complete(request))
    with pytest.raises(RecordingExhaustedError):
        asyncio.run(adapter.complete(request))


# --- LIVE: same config, or a new one ----------------------------------------------


def test_live_reexecution_with_the_same_config_is_never_called_deterministic() -> None:
    run = original_run()
    different_answer = [SCRIPT[0], SCRIPT[1], final("20.40")]
    result = asyncio.run(
        service().reexecute_live(run, CASE, ScriptedModelAdapter(different_answer))
    )
    assert result.mode is ReplayMode.LIVE_SAME_CONFIG
    assert result.deterministic is False
    assert result.difference is not None and result.difference.same_final_answer is False
    assert "not by themselves a bug" in result.note
    assert result.new_run is not None
    assert result.new_run.execution_mode is ExecutionMode.LIVE
    assert result.new_run.agent_config_hash == run.agent_config_hash


def test_live_reexecution_with_a_new_prompt_is_a_new_config() -> None:
    run = original_run()
    v2 = CONFIG.model_copy(
        update={"label": "v2", "prompt": PROMPT.model_copy(update={"version": "2"})}
    )
    result = asyncio.run(
        service().reexecute_live(run, CASE, ScriptedModelAdapter(SCRIPT), new_config=v2)
    )
    assert result.mode is ReplayMode.NEW_CONFIG
    assert result.new_run is not None
    assert result.new_run.agent_config_hash == v2.config_hash != run.agent_config_hash


def test_runs_without_a_model_config_cannot_be_reexecuted() -> None:
    runner = AgentRunner(default_registry(), clock=FakeClock())
    rule_run = asyncio.run(runner.run(CASE.to_task(), RuleBasedAgent()))
    with pytest.raises(ReplayError, match="not made by a configurable model agent"):
        asyncio.run(service().reexecute_with_recorded_responses(rule_run, CASE))


def test_a_tampered_recorded_config_is_rejected() -> None:
    run = original_run()
    started = run.events[0]
    payload = {**started.payload}
    payload["agent"] = {
        **payload["agent"],
        "config": {**payload["agent"]["config"], "model": "other-model"},
    }
    tampered = run.model_copy(
        update={"events": (started.model_copy(update={"payload": payload}), *run.events[1:])}
    )
    with pytest.raises(ReplayError, match="does not match its hash"):
        asyncio.run(service().reexecute_with_recorded_responses(tampered, CASE))


# --- Experiments ------------------------------------------------------------------


def lookup_policy(request: ModelRequest) -> ModelResponse:
    """A fake model that looks the record up, then answers with its quantity."""
    last = request.messages[-1]
    if isinstance(last, ToolResultsMessage):
        quantity = re.search(r'"quantity": (\d+)', last.results[0].content)
        assert quantity is not None
        return final(quantity.group(1))
    first = request.messages[0]
    assert isinstance(first, UserMessage)
    record_id = re.search(r"record ([A-Z]-\d+)", first.text)
    assert record_id is not None
    return tool_use("lookup_record", {"record_id": record_id.group(1)}, "t1")


def guessing_policy(request: ModelRequest) -> ModelResponse:
    return final("100")


def lookup_case(record: Record) -> TestCase:
    return TestCase(
        key=f"qty-{record.id.lower()}",
        prompt=f"What is the quantity of record {record.id}?",
        allowed_tools=("lookup_record",),
        expectations=Expectations(
            final_answer=NumericAnswer(value=record.quantity),
            required_tools=(ToolExpectation(name="lookup_record"),),
        ),
    )


BENCHMARK = Benchmark(
    name="lookups", version="1", cases=tuple(lookup_case(r) for r in DEFAULT_RECORDS[:4])
)


def test_experiment_compares_two_configurations_on_the_same_benchmark() -> None:
    careful = CONFIG.model_copy(update={"label": "careful"})
    hasty = CONFIG.model_copy(
        update={"label": "hasty", "prompt": PROMPT.model_copy(update={"version": "2-terse"})}
    )
    policies = {careful.config_hash: lookup_policy, hasty.config_hash: guessing_policy}

    def adapter_for(config: ModelAgentConfig) -> ScriptedModelAdapter:
        return ScriptedModelAdapter([policies[config.config_hash]] * 5)

    clock = FakeClock()
    runner = BenchmarkRunner(AgentRunner(default_registry(), clock=clock), EvaluationEngine())
    report = asyncio.run(
        ExperimentService(runner, clock=clock).compare(
            BENCHMARK,
            ConfiguredAgent(config=careful, limits=RunLimits()),
            ConfiguredAgent(config=hasty, limits=RunLimits()),
            adapter_for,
        )
    )
    assert report.changed_fields == ("prompt",)
    assert report.baseline_summary.passed == 4
    assert report.candidate_summary.passed == 0
    assert set(report.comparison.broken) == {c.key for c in BENCHMARK.cases}
    success = next(m for m in report.comparison.metrics if m.metric == "task_success_rate")
    assert success.percentage_point_change == -100.0
    assert report.baseline_summary.agent.agent_version == careful.version
    assert LIVE_MODEL_NOTE not in report.notes  # scripted models: no live calls


def test_experiment_with_a_live_provider_carries_the_non_determinism_note() -> None:
    live = CONFIG.model_copy(update={"provider": "anthropic", "label": "live"})

    class FakeLive(ScriptedModelAdapter):
        provider = "anthropic"

    clock = FakeClock()
    runner = BenchmarkRunner(AgentRunner(default_registry(), clock=clock), EvaluationEngine())
    report = asyncio.run(
        ExperimentService(runner, clock=clock).compare(
            BENCHMARK,
            ConfiguredAgent(config=live, limits=RunLimits()),
            ConfiguredAgent(config=live, limits=RunLimits()),
            lambda config: FakeLive([lookup_policy] * 5),
        )
    )
    assert LIVE_MODEL_NOTE in report.notes
    assert any("identical" in note for note in report.notes)
    assert report.comparison.fixed == () and report.comparison.broken == ()


def test_model_unavailable_runs_are_invalid_in_experiments() -> None:
    from agentbench.llm.base import ModelTimeoutError

    clock = FakeClock()
    runner = BenchmarkRunner(AgentRunner(default_registry(), clock=clock), EvaluationEngine())
    report = asyncio.run(
        ExperimentService(runner, clock=clock).compare(
            BENCHMARK,
            ConfiguredAgent(config=CONFIG, limits=RunLimits()),
            ConfiguredAgent(config=CONFIG, limits=RunLimits()),
            lambda config: ScriptedModelAdapter([ModelTimeoutError("down")] * 10),
        )
    )
    assert report.baseline_summary.invalid == 4
    assert report.baseline_summary.task_success_rate.value is None
    assert all(
        c.failure_category is not None and c.failure_category.value == "MODEL_UNAVAILABLE"
        for c in report.baseline_summary.cases
    )
    assert report.comparison.cases[0].change.value == "NOT_COMPARABLE"
    assert TerminationReason.MODEL_UNAVAILABLE.status.value == "FAILED"
