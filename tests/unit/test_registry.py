import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agentbench.domain.errors import (
    InvalidToolDefinitionError,
    ToolArgumentsError,
    ToolExecutionError,
    ToolNotAllowedError,
    ToolTimeoutError,
    UnknownToolError,
)
from agentbench.tools.base import Tool
from agentbench.tools.mock import default_registry
from agentbench.tools.mock.calculator import CalculatorTool
from agentbench.tools.mock.faults import FaultInjectingTool
from agentbench.tools.registry import ToolRegistry

ALL = frozenset({"search_records", "lookup_record", "calculator"})


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LaxArgs(BaseModel):
    pass


class BehaviourTool(Tool):
    """A tool whose `run` behaviour is supplied by the test."""

    description = "test tool"
    args_model = StrictArgs

    def __init__(self, name: str, behaviour: Any) -> None:
        self.name = name
        self.behaviour = behaviour

    async def run(self, args: Any) -> Any:
        return await self.behaviour()


def make_tool(name: str = "probe", args_model: type = StrictArgs) -> Tool:
    async def ok() -> dict[str, int]:
        return {"ok": 1}

    tool = BehaviourTool(name, ok)
    tool.args_model = args_model
    return tool


@pytest.mark.parametrize("name", ["", "Bad", "has space", "1st", "x" * 65])
def test_register_rejects_invalid_names(name: str) -> None:
    with pytest.raises(InvalidToolDefinitionError, match="name"):
        ToolRegistry([make_tool(name)])


def test_register_rejects_duplicates() -> None:
    with pytest.raises(InvalidToolDefinitionError, match="already registered"):
        ToolRegistry([make_tool("probe"), make_tool("probe")])


def test_register_rejects_args_model_that_allows_unknown_fields() -> None:
    with pytest.raises(InvalidToolDefinitionError, match="extra='forbid'"):
        ToolRegistry([make_tool(args_model=LaxArgs)])


def test_register_rejects_args_model_that_is_not_pydantic() -> None:
    with pytest.raises(InvalidToolDefinitionError, match="Pydantic"):
        ToolRegistry([make_tool(args_model=dict)])


def test_schemas_describe_each_allowed_tool() -> None:
    schemas = default_registry().schemas(["lookup_record", "calculator"])
    assert [s["name"] for s in schemas] == ["calculator", "lookup_record"]
    assert schemas[1]["parameters"]["required"] == ["record_id"]


def test_validate_rejects_unknown_tool() -> None:
    with pytest.raises(UnknownToolError):
        default_registry().validate("c1", "delete_record", {}, ALL)


def test_validate_rejects_non_string_tool_name() -> None:
    with pytest.raises(UnknownToolError):
        default_registry().validate("c1", None, {}, ALL)


def test_validate_rejects_tool_outside_allow_list() -> None:
    with pytest.raises(ToolNotAllowedError):
        default_registry().validate("c1", "lookup_record", {"record_id": "A-12"}, {"calculator"})


@pytest.mark.parametrize("arguments", ["A-12", ["A-12"], None, 12])
def test_validate_rejects_arguments_that_are_not_an_object(arguments: object) -> None:
    with pytest.raises(ToolArgumentsError, match="JSON object"):
        default_registry().validate("c1", "lookup_record", arguments, ALL)


@pytest.mark.parametrize(
    ("arguments", "error_type"),
    [
        ({}, "missing"),
        ({"id": "A-12"}, "missing"),  # wrong argument name
        ({"record_id": "A-12", "extra": 1}, "extra_forbidden"),
        ({"record_id": 12}, "string_type"),
        ({"record_id": "not-an-id"}, "string_pattern_mismatch"),
    ],
)
def test_validate_reports_field_level_argument_errors(
    arguments: dict[str, object], error_type: str
) -> None:
    with pytest.raises(ToolArgumentsError) as info:
        default_registry().validate("c1", "lookup_record", arguments, ALL)
    assert error_type in {e["type"] for e in info.value.details["errors"]}


def test_validate_returns_parsed_arguments() -> None:
    call = default_registry().validate("c1", "calculator", {"expression": "1 + 1"}, ALL)
    assert call.args.expression == "1 + 1"


def test_execute_times_out_a_hanging_tool() -> None:
    registry = ToolRegistry([FaultInjectingTool(CalculatorTool(), failures=1, mode="hang")])
    call = registry.validate("c1", "calculator", {"expression": "1"}, ALL)
    with pytest.raises(ToolTimeoutError) as info:
        asyncio.run(registry.execute(call, timeout_s=0.01))
    assert info.value.retryable is True


def test_execute_wraps_unexpected_exceptions_without_hiding_them() -> None:
    async def boom() -> None:
        raise KeyError("missing")

    registry = ToolRegistry([BehaviourTool("probe", boom)])
    call = registry.validate("c1", "probe", {}, frozenset({"probe"}))
    with pytest.raises(ToolExecutionError, match="KeyError") as info:
        asyncio.run(registry.execute(call, timeout_s=1))
    assert info.value.retryable is False


@pytest.mark.parametrize("output", [{1, 2}, {"value": float("nan")}, object()])
def test_execute_rejects_output_that_is_not_json(output: object) -> None:
    async def bad() -> object:
        return output

    registry = ToolRegistry([BehaviourTool("probe", bad)])
    call = registry.validate("c1", "probe", {}, frozenset({"probe"}))
    with pytest.raises(ToolExecutionError, match="not valid JSON"):
        asyncio.run(registry.execute(call, timeout_s=1))
