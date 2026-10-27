"""Provider-independent model interface.

A ModelAdapter turns a ModelRequest (system prompt, conversation, permitted tool
schemas, decoding settings) into a normalised ModelResponse (text and/or proposed tool
calls, stop reason, usage), and turns every provider failure into a ModelError with an
explicit `retryable` flag. Nothing above this layer knows which provider is used.
"""

from typing import Any, Literal, Protocol

from pydantic import Field

from agentbench.domain.agent_config import DecodingSettings
from agentbench.domain.errors import AgentBenchError, AgentInfrastructureError
from agentbench.domain.models import DomainModel, TokenUsage


class ModelToolCall(DomainModel):
    id: str = Field(min_length=1)
    name: str
    arguments: Any  # untrusted: validated by the ToolRegistry, never executed as code


class ToolResultContent(DomainModel):
    call_id: str
    content: str  # JSON text of the tool output or error
    is_error: bool


class UserMessage(DomainModel):
    role: Literal["user"] = "user"
    text: str


class AssistantMessage(DomainModel):
    role: Literal["assistant"] = "assistant"
    text: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    # The provider's own content blocks for this turn (e.g. including thinking blocks),
    # sent back unchanged when present. Some providers reject edited history.
    provider_content: tuple[dict[str, Any], ...] = ()


class ToolResultsMessage(DomainModel):
    role: Literal["tool_results"] = "tool_results"
    results: tuple[ToolResultContent, ...]


Message = UserMessage | AssistantMessage | ToolResultsMessage


class ToolSpec(DomainModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


class ModelRequest(DomainModel):
    model: str
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    decoding: DecodingSettings
    timeout_s: float


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


class ModelResponse(DomainModel):
    text: str | None
    tool_calls: tuple[ModelToolCall, ...]
    stop_reason: StopReason
    raw_stop_reason: str | None = None  # the provider's own value, for debugging
    usage: TokenUsage | None  # None when the provider did not report usage
    response_id: str | None = None
    model: str | None = None  # the model id the provider says it used
    provider_content: tuple[dict[str, Any], ...] = ()  # raw content blocks, for replay


class ModelError(AgentBenchError):
    def __init__(self, message: str, *, retryable: bool, retry_after_s: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_s = retry_after_s


class ModelTimeoutError(ModelError):
    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class ModelRateLimitError(ModelError):
    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message, retryable=True, retry_after_s=retry_after_s)


class ModelProviderError(ModelError):
    """Any other provider failure. Server errors and connection failures are retryable;
    client errors (bad request, authentication, permission) are not."""


class MalformedModelResponseError(ModelError):
    """The provider answered, but the answer cannot be used (e.g. a tool call without a
    name, or stop_reason tool_use with no tool call). The model's fault: not retried."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class ModelUnavailableError(AgentInfrastructureError):
    """The provider kept failing with retryable errors, or refused the request for a
    reason that is not the model's behaviour (authentication, permissions, a bad
    request). The run is ended as MODEL_UNAVAILABLE: infrastructure, not the agent."""


class ModelAdapter(Protocol):
    provider: str

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


def check_response(response: ModelResponse) -> ModelResponse:
    """Structural checks every adapter applies before returning a response."""
    if response.stop_reason == "tool_use" and not response.tool_calls:
        raise MalformedModelResponseError("stop_reason is tool_use but no tool call was returned")
    empty = not response.tool_calls and not (response.text and response.text.strip())
    # max_tokens and refusal are reported by the agent with their own messages.
    if empty and response.stop_reason not in ("max_tokens", "refusal"):
        raise MalformedModelResponseError("response has neither text nor tool calls")
    ids = [c.id for c in response.tool_calls]
    if len(ids) != len(set(ids)):
        raise MalformedModelResponseError(f"duplicate tool call ids: {ids}")
    return response
