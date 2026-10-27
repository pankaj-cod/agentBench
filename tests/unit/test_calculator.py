import asyncio

import pytest
from pydantic import ValidationError

from agentbench.domain.errors import ToolExecutionError
from agentbench.tools.mock.calculator import CalculatorArgs, CalculatorTool


def calculate(expression: str) -> int | float:
    output = asyncio.run(CalculatorTool().run(CalculatorArgs(expression=expression)))
    return output["result"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", 3),
        ("(12 + 8) * 3", 60),
        ("120 * 0.17", 20.4),  # float noise 20.400000000000002 is normalised away
        ("120 * 17 / 100", 20.4),
        ("7 // 2", 3),
        ("7 % 4", 3),
        ("2 ** 10", 1024),
        ("2 ** -1", 0.5),
        ("-(3 - 5)", 2),
        ("1.5 * 2", 3),  # whole-number floats become ints
        ("0.1 + 0.2", 0.3),
    ],
)
def test_valid_arithmetic(expression: str, expected: float) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "x + 1",
        "(1).__class__",
        "'a' * 3",
        "True + 1",
        "1 < 2",
        "[1, 2]",
        "lambda: 1",
        "1 +",
        "2 ^ 3",  # bitwise xor is not allowed
    ],
)
def test_disallowed_syntax_is_rejected_during_validation(expression: str) -> None:
    with pytest.raises(ValidationError):
        CalculatorArgs(expression=expression)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1 / 0", "division by zero"),
        ("5 % 0", "division by zero"),
        ("10 ** 13", "exponent"),
        ("2 ** 0.5", "exponent"),
        ("1e400", "out of range"),
        ("999999999999999 * 999999", "out of range"),
        ("1e15 ** 12", "out of range"),
    ],
)
def test_runtime_errors_are_tool_errors(expression: str, message: str) -> None:
    with pytest.raises(ToolExecutionError, match=message) as info:
        calculate(expression)
    assert info.value.retryable is False


def test_expression_length_is_limited() -> None:
    with pytest.raises(ValidationError):
        CalculatorArgs(expression="1+" * 150 + "1")
