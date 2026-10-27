"""An agent whose decisions come from a language model, through a ModelAdapter.

Each call to `decide` rebuilds the full conversation from the run's history, sends it
with the permitted tool schemas, and turns the model's response into one decision:

- tool calls proposed -> a ToolCallDecision for the FIRST call. AgentBench runs one tool
  call per step; any further calls in the same response are answered with an error
  result ("not executed") when the conversation is rebuilt, so the model can ask again.
  Parallel tool use is also disabled in the request where the provider supports it.
- text and no tool call -> a FinalAnswerDecision.
- cut off at max_output_tokens, or a refusal -> an exception (AGENT_ERROR).

The agent keeps no state between calls: the normalised model response is stored in
each decision's metadata (and so in the trace), and the next request is rebuilt from
it. That makes runs reproducible from their trace and safe to reuse across tasks.

Model output is untrusted. Tool names and arguments go to the ToolRegistry for
validation like any other agent's; nothing the model writes is ever executed as code.
"""

import json
from typing import Any

from agentbench.agents.base import AgentContext
from agentbench.domain.agent_config import ModelAgentConfig, tool_schemas_hash
from agentbench.domain.models import (
    FinalAnswerDecision,
    HistoryStep,
    Observation,
    ToolCallDecision,
)
from agentbench.llm.base import (
    AssistantMessage,
    MalformedModelResponseError,
    Message,
    ModelAdapter,
    ModelError,
    ModelRequest,
    ModelResponse,
    ModelUnavailableError,
    ToolResultContent,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)
from agentbench.tracing.clock import Clock, SystemClock

NOT_EXECUTED = json.dumps(
    {"error": "not executed: AgentBench runs one tool call per step; request it again if needed"}
)


class ModelAgent:
    name = "model"

    def __init__(
        self, config: ModelAgentConfig, adapter: ModelAdapter, *, clock: Clock | None = None
    ) -> None:
        if adapter.provider != config.provider:
            raise ValueError(
                f"config is for provider {config.provider!r}, adapter is {adapter.provider!r}"
            )
        self.config = config
        self.version = config.version
        self._adapter = adapter
        self._clock = clock or SystemClock()

    @property
    def config_hash(self) -> str:
        return self.config.config_hash

    def describe(self) -> dict[str, Any]:
        return {**self.config.model_dump(mode="json"), "config_hash": self.config_hash}

    async def decide(self, context: AgentContext) -> ToolCallDecision | FinalAnswerDecision:
        request = ModelRequest(
            model=self.config.model,
            system=self.config.prompt.system,
            messages=build_conversation(context),
            tools=tuple(ToolSpec.model_validate(t) for t in context.tools),
            decoding=self.config.decoding,
            timeout_s=self.config.request_timeout_s,
        )
        response, attempts = await self._complete_with_retries(request)
        metadata = {
            "model_response": response.model_dump(mode="json"),
            "api_attempts": attempts,
            "tool_schemas_hash": tool_schemas_hash(context.tools),
        }
        if response.tool_calls:
            first, *rest = response.tool_calls
            if rest:
                metadata["unexecuted_tool_calls"] = [c.id for c in rest]
            return ToolCallDecision(
                tool=first.name, arguments=first.arguments, usage=response.usage, metadata=metadata
            )
        if response.stop_reason == "max_tokens":
            raise MalformedModelResponseError(
                "response was cut off at max_output_tokens before a tool call or final answer"
            )
        if response.stop_reason == "refusal":
            raise MalformedModelResponseError(f"model refused the request: {response.text!r}")
        assert response.text is not None  # guaranteed by check_response
        return FinalAnswerDecision(
            answer=response.text.strip(), usage=response.usage, metadata=metadata
        )

    async def _complete_with_retries(self, request: ModelRequest) -> tuple[ModelResponse, int]:
        """Call the provider, retrying retryable errors with bounded exponential backoff
        (or the provider's retry-after hint). Malformed responses are the model's fault
        and are raised as they are; everything else that persists becomes
        ModelUnavailableError (infrastructure)."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._adapter.complete(request), attempt
            except MalformedModelResponseError:
                raise
            except ModelError as exc:
                if not exc.retryable:
                    raise ModelUnavailableError(
                        f"{type(exc).__name__} (not retryable): {exc}"
                    ) from exc
                if attempt > self.config.max_api_retries:
                    raise ModelUnavailableError(
                        f"{type(exc).__name__} on all {attempt} attempts: {exc}"
                    ) from exc
                backoff = self.config.api_retry_backoff_s * 2 ** (attempt - 1)
                await self._clock.sleep(exc.retry_after_s if exc.retry_after_s else backoff)


def build_conversation(context: AgentContext) -> tuple[Message, ...]:
    """The conversation so far, rebuilt from the run history alone."""
    prompt = context.task.prompt
    if context.task.context:
        prompt += "\n\nContext:\n" + json.dumps(context.task.context, sort_keys=True)
    messages: list[Message] = [UserMessage(text=prompt)]
    for step in context.history:
        messages += _replay_step(step)
    return tuple(messages)


def _replay_step(step: HistoryStep) -> list[Message]:
    recorded = step.decision.metadata.get("model_response")
    if recorded is None:
        raise ValueError(f"step {step.step} has no recorded model response to rebuild from")
    response = ModelResponse.model_validate(recorded)
    if not response.tool_calls or step.observation is None:
        raise ValueError(f"step {step.step} has no tool call to answer")
    first, *rest = response.tool_calls
    results = [_result(first.id, step.observation)]
    results += [ToolResultContent(call_id=c.id, content=NOT_EXECUTED, is_error=True) for c in rest]
    return [
        AssistantMessage(
            text=response.text,
            tool_calls=response.tool_calls,
            provider_content=response.provider_content,
        ),
        ToolResultsMessage(results=tuple(results)),
    ]


def _result(call_id: str, observation: Observation) -> ToolResultContent:
    if observation.ok:
        return ToolResultContent(
            call_id=call_id, content=json.dumps(observation.output), is_error=False
        )
    assert observation.error is not None
    error = observation.error
    content = {"error": {"type": error.type, "message": error.message, "details": error.details}}
    return ToolResultContent(call_id=call_id, content=json.dumps(content), is_error=True)
