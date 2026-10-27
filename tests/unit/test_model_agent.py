"""ModelAgent + AgentRunner with a scripted (fake) model: no network, no API keys."""

import asyncio
import json
from typing import Any

import pytest

from agentbench.agents.model_agent import NOT_EXECUTED, ModelAgent
from agentbench.domain.agent_config import DecodingSettings, ModelAgentConfig, PromptTemplate
from agentbench.domain.benchmark import Expectations, NumericAnswer, TestCase, ToolExpectation
from agentbench.domain.enums import TerminationReason
from agentbench.domain.evaluation import EvaluationOutcome, FailureCategory
from agentbench.domain.models import RunLimits, RunResult, Task, TokenUsage
from agentbench.evaluation.engine import EvaluationEngine
from agentbench.llm.base import (
    AssistantMessage,
    ModelProviderError,
    ModelRateLimitError,
    ModelResponse,
    ModelTimeoutError,
    ModelToolCall,
    ToolResultsMessage,
    UserMessage,
)
from agentbench.llm.scripted import ScriptedModelAdapter
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import default_registry
from agentbench.tracing.clock import FakeClock

TOOLS = ("search_records", "lookup_record", "calculator")
TASK = Task(key="pct", prompt="What is 17% of the quantity of record A-12?", allowed_tools=TOOLS)
CONFIG = ModelAgentConfig(
    label="test",
    provider="scripted",
    model="fake-model-1",
    prompt=PromptTemplate(id="inventory", version="1", system="Use the tools."),
    api_retry_backoff_s=0.5,
)
USAGE = TokenUsage(model="fake-model-1", input_tokens=100, output_tokens=10)


def call(name: str, arguments: Any, call_id: str = "t1") -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=arguments)


def tool_use(*calls: ModelToolCall, usage: TokenUsage | None = USAGE) -> ModelResponse:
    return ModelResponse(text=None, tool_calls=calls, stop_reason="tool_use", usage=usage)


def final(text: str, usage: TokenUsage | None = USAGE) -> ModelResponse:
    return ModelResponse(text=text, tool_calls=(), stop_reason="end_turn", usage=usage)


def run(
    script: list[Any], limits: RunLimits | None = None, config: ModelAgentConfig = CONFIG
) -> tuple[RunResult, ScriptedModelAdapter, FakeClock]:
    clock = FakeClock()
    adapter = ScriptedModelAdapter(script)
    agent = ModelAgent(config, adapter, clock=clock)
    result = asyncio.run(AgentRunner(default_registry(), clock=clock).run(TASK, agent, limits))
    return result, adapter, clock


def event_types(result: RunResult) -> list[str]:
    return [e.type.value for e in result.events]


GOOD_SCRIPT = [
    tool_use(call("lookup_record", {"record_id": "A-12"}, "t1")),
    tool_use(call("calculator", {"expression": "120 * 17 / 100"}, "t2")),
    final("20.4"),
]


def test_full_tool_loop_with_usage() -> None:
    result, adapter, _ = run(GOOD_SCRIPT)
    assert result.termination_reason is TerminationReason.COMPLETED
    assert result.final_answer == "20.4"
    assert (result.input_tokens, result.output_tokens) == (300, 30)
    assert len(adapter.requests) == 3


def test_conversation_is_rebuilt_with_real_tool_results() -> None:
    _, adapter, _ = run(GOOD_SCRIPT)
    third = adapter.requests[2].messages
    assert [m.role for m in third] == [
        "user",
        "assistant",
        "tool_results",
        "assistant",
        "tool_results",
    ]
    first_results = third[2]
    assert isinstance(first_results, ToolResultsMessage)
    assert first_results.results[0].call_id == "t1"
    assert json.loads(first_results.results[0].content)["record"]["quantity"] == 120
    assert first_results.results[0].is_error is False


def test_only_permitted_tool_schemas_are_sent() -> None:
    task = Task(key="calc", prompt="Compute 1 + 1", allowed_tools=("calculator",))
    adapter = ScriptedModelAdapter([final("2")])
    agent = ModelAgent(CONFIG, adapter, clock=FakeClock())
    asyncio.run(AgentRunner(default_registry(), clock=FakeClock()).run(task, agent))
    assert [t.name for t in adapter.requests[0].tools] == ["calculator"]


def test_unknown_tool_is_rejected_and_reported_back_to_the_model() -> None:
    result, adapter, _ = run([tool_use(call("delete_all_records", {})), final("cannot do that")])
    assert event_types(result) == [
        "RUN_STARTED", "AGENT_DECISION", "VALIDATION_ERROR", "AGENT_DECISION", "FINAL_ANSWER",
        "RUN_FINISHED",
    ]  # fmt: skip
    feedback = adapter.requests[1].messages[2]
    assert isinstance(feedback, ToolResultsMessage)
    assert feedback.results[0].is_error is True
    assert json.loads(feedback.results[0].content)["error"]["type"] == "UnknownToolError"


@pytest.mark.parametrize(
    "arguments",
    ["A-12", ["A-12"], {"id": "A-12"}, {"record_id": "A-12; DROP TABLE"}, {"record_id": 12}],
)
def test_malformed_tool_arguments_never_reach_the_tool(arguments: Any) -> None:
    result, adapter, _ = run([tool_use(call("lookup_record", arguments)), final("gave up")])
    assert "TOOL_CALL_STARTED" not in event_types(result)
    assert "VALIDATION_ERROR" in event_types(result)
    feedback = adapter.requests[1].messages[2]
    assert json.loads(feedback.results[0].content)["error"]["type"] == "ToolArgumentsError"


def test_code_in_calculator_arguments_is_rejected_not_executed() -> None:
    malicious = {"expression": "__import__('os').system('echo pwned')"}
    result, _, _ = run([tool_use(call("calculator", malicious)), final("x")])
    assert "TOOL_CALL_STARTED" not in event_types(result)


def test_repeated_identical_calls_end_at_the_step_limit() -> None:
    script = [tool_use(call("lookup_record", {"record_id": "A-12"}, f"t{i}")) for i in range(10)]
    result, adapter, _ = run(script, RunLimits(max_steps=4))
    assert result.termination_reason is TerminationReason.MAX_STEPS_EXCEEDED
    assert len(adapter.requests) == 4
    assert result.tool_call_count == 4


def test_extra_parallel_tool_calls_are_not_executed_and_the_model_is_told() -> None:
    both = tool_use(
        call("lookup_record", {"record_id": "A-12"}, "t1"),
        call("lookup_record", {"record_id": "A-15"}, "t2"),
    )
    result, adapter, _ = run([both, final("120")])
    started = [e for e in result.events if e.type == "TOOL_CALL_STARTED"]
    assert len(started) == 1
    decision = next(e for e in result.events if e.type == "AGENT_DECISION")
    assert decision.payload["metadata"]["unexecuted_tool_calls"] == ["t2"]
    feedback = adapter.requests[1].messages[2]
    assert isinstance(feedback, ToolResultsMessage)
    results = feedback.results
    assert [(r.call_id, r.is_error) for r in results] == [("t1", False), ("t2", True)]
    assert results[1].content == NOT_EXECUTED


@pytest.mark.parametrize(
    "bad",
    [
        ModelResponse(text=None, tool_calls=(), stop_reason="tool_use", usage=None),
        ModelResponse(text="   ", tool_calls=(), stop_reason="end_turn", usage=None),
        ModelResponse(
            text=None,
            tool_calls=(call("calculator", {}, "same"), call("calculator", {}, "same")),
            stop_reason="tool_use",
            usage=None,
        ),
    ],
    ids=["tool_use-without-calls", "empty-text", "duplicate-call-ids"],
)
def test_malformed_responses_are_agent_errors(bad: ModelResponse) -> None:
    result, _, _ = run([bad])
    assert result.termination_reason is TerminationReason.AGENT_ERROR
    assert "MalformedModelResponseError" in (result.error_message or "")


def test_truncated_response_is_an_agent_error() -> None:
    cut = ModelResponse(text="The answer is", tool_calls=(), stop_reason="max_tokens", usage=USAGE)
    result, _, _ = run([cut])
    assert result.termination_reason is TerminationReason.AGENT_ERROR
    assert "max_output_tokens" in (result.error_message or "")


def test_timeouts_are_retried_with_backoff_then_succeed() -> None:
    script = [ModelTimeoutError("t"), ModelTimeoutError("t"), *GOOD_SCRIPT]
    result, _, clock = run(script)
    assert result.termination_reason is TerminationReason.COMPLETED
    assert clock.sleeps == [0.5, 1.0]
    first = next(e for e in result.events if e.type == "AGENT_DECISION")
    assert first.payload["metadata"]["api_attempts"] == 3


def test_persistent_timeouts_make_the_model_unavailable_not_an_agent_failure() -> None:
    result, adapter, _ = run([ModelTimeoutError("timed out")] * 3)
    assert result.termination_reason is TerminationReason.MODEL_UNAVAILABLE
    assert len(adapter.requests) == 3  # 1 + max_api_retries
    case = TestCase(
        key="pct",
        prompt=TASK.prompt,
        allowed_tools=TOOLS,
        expectations=Expectations(final_answer=NumericAnswer(value=20.4, tolerance=0.01)),
    )
    evaluation = EvaluationEngine().evaluate(case, result)
    assert evaluation.outcome is EvaluationOutcome.INVALID
    assert evaluation.failure_category is FailureCategory.MODEL_UNAVAILABLE


def test_rate_limit_retry_after_hint_is_used() -> None:
    result, _, clock = run([ModelRateLimitError("slow down", retry_after_s=7), *GOOD_SCRIPT])
    assert result.termination_reason is TerminationReason.COMPLETED
    assert clock.sleeps == [7]


def test_non_retryable_provider_error_is_not_retried() -> None:
    error = ModelProviderError("401 invalid x-api-key", retryable=False)
    result, adapter, clock = run([error, *GOOD_SCRIPT])
    assert result.termination_reason is TerminationReason.MODEL_UNAVAILABLE
    assert len(adapter.requests) == 1 and clock.sleeps == []


def test_missing_usage_metadata_makes_run_totals_unavailable() -> None:
    script = [
        tool_use(call("lookup_record", {"record_id": "A-12"}), usage=None),
        tool_use(call("calculator", {"expression": "120 * 17 / 100"}, "t2")),
        final("20.4"),
    ]
    result, _, _ = run(script)
    assert result.termination_reason is TerminationReason.COMPLETED
    assert (result.input_tokens, result.output_tokens) == (None, None)


def test_versioning_is_recorded_in_the_trace_and_result() -> None:
    result, _, _ = run(GOOD_SCRIPT)
    started = result.events[0].payload["agent"]
    assert started["version"] == CONFIG.version
    assert started["config"]["model"] == "fake-model-1"
    assert started["config"]["config_hash"] == CONFIG.config_hash
    assert result.agent_config_hash == CONFIG.config_hash
    assert result.tool_schemas_hash is not None
    assert result.events[-1].payload["tool_schemas_hash"] == result.tool_schemas_hash


def test_config_hash_covers_behaviour_but_not_the_label() -> None:
    relabelled = CONFIG.model_copy(update={"label": "other name"})
    new_prompt = CONFIG.model_copy(
        update={"prompt": PromptTemplate(id="inventory", version="2", system="Use the tools.")}
    )
    new_decoding = CONFIG.model_copy(update={"decoding": DecodingSettings(temperature=0.0)})
    new_model = CONFIG.model_copy(update={"model": "fake-model-2"})
    assert relabelled.config_hash == CONFIG.config_hash
    hashes = {c.config_hash for c in (CONFIG, new_prompt, new_decoding, new_model)}
    assert len(hashes) == 4


def test_adapter_must_match_the_configured_provider() -> None:
    class Other(ScriptedModelAdapter):
        provider = "anthropic"

    with pytest.raises(ValueError, match="provider"):
        ModelAgent(CONFIG, Other([]))


def test_first_request_contains_only_the_task_prompt() -> None:
    _, adapter, _ = run(GOOD_SCRIPT)
    first = adapter.requests[0]
    assert len(first.messages) == 1
    assert isinstance(first.messages[0], UserMessage)
    assert first.messages[0].text == TASK.prompt
    assert first.system == "Use the tools."
    assert isinstance(adapter.requests[1].messages[1], AssistantMessage)


def test_required_tool_expectations_work_for_model_runs() -> None:
    result, _, _ = run(GOOD_SCRIPT)
    case = TestCase(
        key="pct",
        prompt=TASK.prompt,
        allowed_tools=TOOLS,
        expectations=Expectations(
            final_answer=NumericAnswer(value=20.4, tolerance=0.01),
            required_tools=(
                ToolExpectation(name="lookup_record", args_subset={"record_id": "A-12"}),
            ),
        ),
    )
    assert EvaluationEngine().evaluate(case, result).outcome is EvaluationOutcome.PASSED
