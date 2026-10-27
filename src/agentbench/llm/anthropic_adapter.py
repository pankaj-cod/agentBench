"""ModelAdapter for Anthropic's Messages API, using the official `anthropic` SDK.

Install with `pip install agentbench[anthropic]`. Credentials are resolved by the SDK
from the environment (ANTHROPIC_API_KEY, or another configured credential source);
this module never reads, stores, or logs a key.

Behaviour:
- Tools are sent with `tool_choice: auto` and parallel tool use disabled.
- The SDK's own retries are turned off (`max_retries=0`); ModelAgent does bounded
  retries itself, so every attempt is visible in the trace.
- Assistant turns are sent back using the provider's raw content blocks when they
  were recorded, unchanged, so thinking blocks and edits-sensitive history survive.
- No sampling parameters are sent (current models reject them). Refusal fallbacks to
  another model are NOT enabled: in a benchmark the configured model must be the one
  that answers. `ModelResponse.model` records which model actually did.
"""

from typing import Any

from agentbench.domain.models import TokenUsage
from agentbench.llm.base import (
    AssistantMessage,
    ModelProviderError,
    ModelRateLimitError,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
    ModelToolCall,
    StopReason,
    ToolResultsMessage,
    UserMessage,
    check_response,
)

STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}
RETRYABLE_STATUS = {408, 409, 429}


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, client: Any = None) -> None:
        import anthropic  # optional dependency: only needed when this adapter is used

        self._anthropic = anthropic
        self._client = client if client is not None else anthropic.AsyncAnthropic(max_retries=0)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        anthropic = self._anthropic
        try:
            message = await self._client.messages.create(
                **to_anthropic_params(request), timeout=request.timeout_s
            )
        except anthropic.APITimeoutError as exc:  # subclass of APIConnectionError
            raise ModelTimeoutError(f"request timed out after {request.timeout_s}s") from exc
        except anthropic.RateLimitError as exc:
            raise ModelRateLimitError(
                f"rate limited: {exc.message}", retry_after_s=_retry_after(exc)
            ) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code >= 500 or exc.status_code in RETRYABLE_STATUS
            raise ModelProviderError(
                f"HTTP {exc.status_code}: {exc.message}", retryable=retryable
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ModelProviderError(f"connection error: {exc}", retryable=True) from exc
        return check_response(from_anthropic_message(message, request.model))


def to_anthropic_params(request: ModelRequest) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.decoding.max_output_tokens,
        "system": request.system,
        "messages": [_message(m) for m in request.messages],
    }
    if request.tools:
        params["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in request.tools
        ]
        params["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    if request.decoding.stop_sequences:
        params["stop_sequences"] = list(request.decoding.stop_sequences)
    if request.decoding.effort is not None:
        params["output_config"] = {"effort": request.decoding.effort}
    return params


def _message(message: UserMessage | AssistantMessage | ToolResultsMessage) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.text}
    if isinstance(message, AssistantMessage):
        if message.provider_content:
            return {"role": "assistant", "content": [dict(b) for b in message.provider_content]}
        blocks: list[dict[str, Any]] = []
        if message.text:
            blocks.append({"type": "text", "text": message.text})
        blocks += [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in message.tool_calls
        ]
        return {"role": "assistant", "content": blocks}
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in message.results
        ],
    }


def from_anthropic_message(message: Any, requested_model: str) -> ModelResponse:
    texts = [b.text for b in message.content if b.type == "text"]
    calls = tuple(
        ModelToolCall(id=b.id, name=b.name, arguments=b.input)
        for b in message.content
        if b.type == "tool_use"
    )
    usage = None
    if message.usage is not None and message.usage.input_tokens is not None:
        usage = TokenUsage(
            model=message.model or requested_model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
    return ModelResponse(
        text="".join(texts) if texts else None,
        tool_calls=calls,
        stop_reason=STOP_REASONS.get(message.stop_reason or "", "other"),
        raw_stop_reason=message.stop_reason,
        usage=usage,
        response_id=message.id,
        model=message.model,
        provider_content=tuple(b.to_dict() for b in message.content),
    )


def _retry_after(exc: Any) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None
