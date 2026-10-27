from agentbench.agents.model_agent import request_hash
from agentbench.domain.enums import TraceEventType
from agentbench.domain.models import RunResult
from agentbench.llm.base import ModelRequest, ModelResponse, ModelUnavailableError


class RecordingExhaustedError(ModelUnavailableError):
    """The re-execution asked the model more often than the original run did."""


class RecordedResponseAdapter:
    """Answers with the model responses stored in an earlier run's trace, in order,
    without calling any provider.

    It also compares each new request with the hash recorded for the original request.
    A mismatch means the conversation changed (usually because a tool returned
    something different); it is recorded in `divergent_steps`, not hidden.
    """

    def __init__(self, source: RunResult, provider: str) -> None:
        self.provider = provider
        decisions = [e for e in source.events if e.type is TraceEventType.AGENT_DECISION]
        self._responses = [
            ModelResponse.model_validate(e.payload["metadata"]["model_response"]) for e in decisions
        ]
        self._hashes = [e.payload["metadata"].get("request_hash") for e in decisions]
        self.calls = 0
        self.divergent_steps: list[int] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        index = self.calls
        self.calls += 1
        if index >= len(self._responses):
            raise RecordingExhaustedError(
                f"the original run recorded {len(self._responses)} model responses; "
                f"the re-execution asked for response {index + 1}"
            )
        if self._hashes[index] is not None and self._hashes[index] != request_hash(request):
            self.divergent_steps.append(index + 1)
        return self._responses[index]
