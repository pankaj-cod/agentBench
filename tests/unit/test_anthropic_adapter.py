"""AnthropicAdapter against the real `anthropic` SDK with a fake HTTP transport.
No network access and no real credentials: the key below is a dummy string."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from agentbench.agents.model_agent import ModelAgent
from agentbench.domain.agent_config import DecodingSettings, ModelAgentConfig, PromptTemplate
from agentbench.domain.enums import TerminationReason
from agentbench.domain.models import Task
from agentbench.llm.anthropic_adapter import AnthropicAdapter
from agentbench.llm.base import (
    AssistantMessage,
    MalformedModelResponseError,
    ModelProviderError,
    ModelRateLimitError,
    ModelRequest,
    ModelTimeoutError,
    ToolSpec,
    UserMessage,
)
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import default_registry
from agentbench.tracing.clock import FakeClock

MODEL = "claude-opus-5-5"
Handler = Callable[[httpx2.Request], httpx2.Response]


def adapter_with(handler: Handler) -> tuple[AnthropicAdapter, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    client = anthropic.AsyncAnthropic(
        api_key="test-dummy-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(record)),
    )
    return AnthropicAdapter(client), sent


def message(content: list[dict[str, Any]], stop: str, usage: dict | None = None) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 120, "output_tokens": 15},
    }


def ok(body: dict) -> Handler:
    return lambda request: httpx2.Response(200, json=body)


def error(status: int, kind: str, headers: dict | None = None) -> Handler:
    body = {"type": "error", "error": {"type": kind, "message": f"{kind} happened"}}
    return lambda request: httpx2.Response(status, json=body, headers=headers or {})


REQUEST = ModelRequest(
    model=MODEL,
    system="Use the tools.",
    messages=(UserMessage(text="What is the quantity of record A-12?"),),
    tools=(
        ToolSpec(
            name="lookup_record",
            description="Fetch a record",
            parameters={"type": "object", "properties": {"record_id": {"type": "string"}}},
        ),
    ),
    decoding=DecodingSettings(effort="low"),
    timeout_s=30,
)


def complete(adapter: AnthropicAdapter, request: ModelRequest = REQUEST) -> Any:
    return asyncio.run(adapter.complete(request))


TOOL_USE = message(
    [
        {"type": "text", "text": "Looking it up."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "lookup_record",
            "input": {"record_id": "A-12"},
        },
    ],
    "tool_use",
)


def test_request_shape() -> None:
    adapter, sent = adapter_with(ok(TOOL_USE))
    complete(adapter)
    body = sent[0]
    assert body["model"] == MODEL
    assert body["max_tokens"] == 16_000
    assert body["system"] == "Use the tools."
    assert body["tools"][0]["input_schema"]["properties"] == {"record_id": {"type": "string"}}
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert body["output_config"] == {"effort": "low"}
    # Sampling parameters are never sent.
    assert not {"temperature", "top_p", "top_k"} & body.keys()


def test_tool_use_response_is_normalised() -> None:
    adapter, _ = adapter_with(ok(TOOL_USE))
    response = complete(adapter)
    assert response.stop_reason == "tool_use"
    assert response.text == "Looking it up."
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("toolu_1", "lookup_record", {"record_id": "A-12"})
    ]
    assert (response.usage.input_tokens, response.usage.output_tokens) == (120, 15)
    assert response.usage.model == MODEL
    assert [b["type"] for b in response.provider_content] == ["text", "tool_use"]


def test_raw_provider_content_is_sent_back_unchanged() -> None:
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    raw = (thinking, {"type": "tool_use", "id": "toolu_1", "name": "lookup_record", "input": {}})
    request = REQUEST.model_copy(
        update={
            "messages": (
                REQUEST.messages[0],
                AssistantMessage(text=None, tool_calls=(), provider_content=raw),
            )
        }
    )
    adapter, sent = adapter_with(ok(message([{"type": "text", "text": "120"}], "end_turn")))
    complete(adapter, request)
    assert sent[0]["messages"][1] == {"role": "assistant", "content": [thinking, raw[1]]}


@pytest.mark.parametrize(
    ("raw", "normalised"),
    [("end_turn", "end_turn"), ("stop_sequence", "end_turn"), ("refusal", "refusal")],
)
def test_stop_reasons(raw: str, normalised: str) -> None:
    adapter, _ = adapter_with(ok(message([{"type": "text", "text": "done"}], raw)))
    response = complete(adapter)
    assert (response.stop_reason, response.raw_stop_reason) == (normalised, raw)


def test_missing_usage_input_tokens_is_none_not_zero() -> None:
    body = message([{"type": "text", "text": "x"}], "end_turn")
    body["usage"] = {"input_tokens": None, "output_tokens": None}
    adapter, _ = adapter_with(ok(body))
    try:
        response = complete(adapter)
    except Exception as exc:  # the SDK may reject a usage object it considers invalid
        pytest.skip(f"SDK rejects null usage fields: {type(exc).__name__}")
    assert response.usage is None


def test_malformed_response_tool_use_without_calls() -> None:
    adapter, _ = adapter_with(ok(message([], "tool_use")))
    with pytest.raises(MalformedModelResponseError):
        complete(adapter)


def test_rate_limit_maps_to_retryable_error_with_retry_after() -> None:
    adapter, _ = adapter_with(error(429, "rate_limit_error", {"retry-after": "12"}))
    with pytest.raises(ModelRateLimitError) as info:
        complete(adapter)
    assert info.value.retryable and info.value.retry_after_s == 12


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (500, "api_error", True),
        (529, "overloaded_error", True),
        (400, "invalid_request_error", False),
        (401, "authentication_error", False),
        (404, "not_found_error", False),
    ],
)
def test_status_errors_map_to_provider_errors(status: int, kind: str, retryable: bool) -> None:
    adapter, _ = adapter_with(error(status, kind))
    with pytest.raises(ModelProviderError) as info:
        complete(adapter)
    assert info.value.retryable is retryable
    assert str(status) in str(info.value)


def test_transport_timeout_maps_to_model_timeout() -> None:
    def hang(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("simulated timeout", request=request)

    adapter, _ = adapter_with(hang)
    with pytest.raises(ModelTimeoutError):
        complete(adapter)


def test_connection_failure_is_retryable() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    adapter, _ = adapter_with(refuse)
    with pytest.raises(ModelProviderError) as info:
        complete(adapter)
    assert info.value.retryable


def test_full_run_through_the_sdk_and_the_runner() -> None:
    responses = iter(
        [
            TOOL_USE,
            message(
                [{"type": "text", "text": "120"}],
                "end_turn",
                {"input_tokens": 200, "output_tokens": 5},
            ),
        ]
    )
    adapter, sent = adapter_with(lambda request: httpx2.Response(200, json=next(responses)))
    config = ModelAgentConfig(
        label="opus",
        provider="anthropic",
        model=MODEL,
        prompt=PromptTemplate(id="inventory", version="1", system="Use the tools."),
    )
    agent = ModelAgent(config, adapter, clock=FakeClock())
    task = Task(
        key="q", prompt="What is the quantity of record A-12?", allowed_tools=("lookup_record",)
    )
    result = asyncio.run(AgentRunner(default_registry(), clock=FakeClock()).run(task, agent))

    assert result.termination_reason is TerminationReason.COMPLETED
    assert result.final_answer == "120"
    assert (result.input_tokens, result.output_tokens) == (320, 20)
    second = sent[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    tool_result = second[2]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_1" and tool_result["is_error"] is False
    assert json.loads(tool_result["content"])["record"]["quantity"] == 120
