"""Arithmetic without eval.

The expression is parsed into a Python syntax tree, and only an explicit allow-list of
node types is accepted: number literals, + - * / // % **, unary + -, and parentheses.
Names, calls, attribute access, strings, comparisons and everything else are rejected
during argument validation, before the tool runs.
"""

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentbench.domain.errors import ToolExecutionError
from agentbench.tools.base import Tool

MAX_ABS_VALUE = 1e15
MAX_EXPONENT = 12

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def parse_expression(expression: str) -> ast.expr:
    """Parse and check that the expression uses only allowed syntax."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid arithmetic expression: {exc.msg}") from None
    for node in ast.walk(tree.body):
        if isinstance(node, ast.Constant):
            # bool is a subclass of int, so check it explicitly.
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ValueError(f"only numbers are allowed, got {node.value!r}")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY:
                raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        elif not isinstance(node, ast.operator | ast.unaryop):
            raise ValueError(f"syntax not allowed: {type(node).__name__}")
    return tree.body


def _evaluate(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        return _checked(node.value)
    if isinstance(node, ast.UnaryOp):
        return _checked(_UNARY[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (
            abs(right) > MAX_EXPONENT or not float(right).is_integer()
        ):
            raise ToolExecutionError(
                f"exponent must be a whole number with |exponent| <= {MAX_EXPONENT}"
            )
        try:
            return _checked(_BINARY[type(node.op)](left, right))
        except ZeroDivisionError:
            raise ToolExecutionError("division by zero") from None
        except OverflowError:
            raise ToolExecutionError("result out of range") from None
    raise AssertionError(f"unvalidated node reached the evaluator: {type(node).__name__}")


def _checked(value: float) -> float:
    if isinstance(value, complex) or not math.isfinite(value) or abs(value) > MAX_ABS_VALUE:
        raise ToolExecutionError(f"result out of range (|value| must be <= {MAX_ABS_VALUE:g})")
    return value


def normalise(value: float) -> int | float:
    """Whole numbers become ints; other floats are rounded to 10 decimal places so that
    float noise such as 20.400000000000002 does not leak into answers."""
    if float(value).is_integer():
        return int(value)
    return round(value, 10)


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expression: str = Field(min_length=1, max_length=200, description="e.g. (12 + 8) * 3")

    @field_validator("expression")
    @classmethod
    def _allowed_syntax_only(cls, value: str) -> str:
        parse_expression(value)
        return value


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression with numbers, + - * / // % ** and ()."
    args_model = CalculatorArgs

    async def run(self, args: CalculatorArgs) -> dict[str, Any]:
        result = _evaluate(parse_expression(args.expression))
        return {"expression": args.expression, "result": normalise(result)}
