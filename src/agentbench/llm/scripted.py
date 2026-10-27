from collections.abc import Callable, Sequence

from agentbench.llm.base import ModelRequest, ModelResponse, check_response

ScriptItem = ModelResponse | Exception | Callable[[ModelRequest], ModelResponse]


class ScriptedModelAdapter:
    """A fake model for tests and offline use: returns (or raises) pre-written items in
    order and keeps every request it received. Never touches the network."""

    provider = "scripted"

    def __init__(self, script: Sequence[ScriptItem]) -> None:
        self._script = list(script)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._script):
            raise AssertionError(
                f"scripted model has {len(self._script)} responses, call {index + 1}"
            )
        item = self._script[index]
        if isinstance(item, Exception):
            raise item
        response = item(request) if callable(item) else item
        return check_response(response)
