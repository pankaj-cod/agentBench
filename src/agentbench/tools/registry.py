import asyncio
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agentbench.domain.errors import (
    InvalidToolDefinitionError,
    ToolArgumentsError,
    ToolError,
    ToolExecutionError,
    ToolNotAllowedError,
    ToolTimeoutError,
    UnknownToolError,
)
from agentbench.tools.base import Tool

TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class ValidatedToolCall:
    call_id: str
    tool: Tool
    args: BaseModel


class ToolRegistry:
    """The catalogue of tools, and the gate every tool call passes through.

    Agent output is untrusted: `validate` rejects unknown tools, tools outside the
    task's allow-list, and arguments that do not match the tool's schema, before
    anything runs. `execute` enforces a timeout and turns every failure into a
    ToolError so the runner never sees an unexpected exception from a tool.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not TOOL_NAME_PATTERN.match(name):
            raise InvalidToolDefinitionError(f"invalid tool name: {name!r}")
        if name in self._tools:
            raise InvalidToolDefinitionError(f"tool already registered: {name}")
        if not getattr(tool, "description", None):
            raise InvalidToolDefinitionError(f"tool {name} has no description")
        args_model = getattr(tool, "args_model", None)
        if not (isinstance(args_model, type) and issubclass(args_model, BaseModel)):
            raise InvalidToolDefinitionError(f"tool {name}: args_model must be a Pydantic model")
        if args_model.model_config.get("extra") != "forbid":
            raise InvalidToolDefinitionError(
                f"tool {name}: args_model must set extra='forbid' so unknown arguments fail"
            )
        self._tools[name] = tool

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(names)]

    def validate(
        self, call_id: str, tool_name: object, arguments: object, allowed: frozenset[str]
    ) -> ValidatedToolCall:
        if not isinstance(tool_name, str) or tool_name not in self._tools:
            raise UnknownToolError(
                f"unknown tool: {tool_name!r}", details={"available": sorted(allowed)}
            )
        if tool_name not in allowed:
            raise ToolNotAllowedError(
                f"tool {tool_name} is not allowed for this task",
                details={"allowed": sorted(allowed)},
            )
        if not isinstance(arguments, Mapping):
            raise ToolArgumentsError(
                f"arguments for {tool_name} must be a JSON object, got {type(arguments).__name__}"
            )
        tool = self._tools[tool_name]
        try:
            args = tool.args_model.model_validate(dict(arguments))
        except ValidationError as exc:
            errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ]
            raise ToolArgumentsError(
                f"invalid arguments for {tool_name}", details={"errors": errors}
            ) from exc
        return ValidatedToolCall(call_id=call_id, tool=tool, args=args)

    async def execute(self, call: ValidatedToolCall, timeout_s: float) -> Any:
        try:
            output = await asyncio.wait_for(call.tool.run(call.args), timeout=timeout_s)
        except TimeoutError as exc:
            raise ToolTimeoutError(
                f"{call.tool.name} did not finish within {timeout_s}s",
                details={"timeout_s": timeout_s},
            ) from exc
        except ToolError:
            raise
        except Exception as exc:
            # A bug or unexpected failure inside the tool: report it, never hide it.
            raise ToolExecutionError(
                f"{call.tool.name} raised {type(exc).__name__}: {exc}", retryable=False
            ) from exc
        try:
            json.dumps(output, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(
                f"{call.tool.name} returned output that is not valid JSON: {exc}",
                retryable=False,
            ) from exc
        return output
