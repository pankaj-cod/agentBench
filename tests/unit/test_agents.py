import asyncio

import pytest

from agentbench.agents.base import AgentContext
from agentbench.agents.rule_based import RuleBasedAgent, UnsupportedTaskError
from agentbench.agents.scripted import ScriptedAgent, ScriptExhaustedError
from agentbench.domain.models import (
    FinalAnswerDecision,
    HistoryStep,
    Observation,
    Task,
    ToolCallDecision,
    ToolErrorInfo,
)


def context(prompt: str, history: tuple[HistoryStep, ...] = ()) -> AgentContext:
    return AgentContext(
        task=Task(key="t", prompt=prompt), step=len(history) + 1, tools=(), history=history
    )


def decide(prompt: str, history: tuple[HistoryStep, ...] = ()) -> object:
    return asyncio.run(RuleBasedAgent().decide(context(prompt, history)))


def ok_step(tool: str, output: object) -> HistoryStep:
    return HistoryStep(
        step=1,
        decision=ToolCallDecision(tool=tool, arguments={}),
        observation=Observation(call_id="call-1", tool=tool, ok=True, output=output),
    )


@pytest.mark.parametrize(
    ("prompt", "tool", "arguments"),
    [
        ("What is the name of record c-03?", "lookup_record", {"record_id": "C-03"}),
        ("What is 50% of the price of record B-09", "lookup_record", {"record_id": "B-09"}),
        ('Find records matching "mouse"', "search_records", {"query": "mouse", "limit": 20}),
        (
            "What is the total quantity of records matching 'steel'?",
            "search_records",
            {"query": "steel", "limit": 20},
        ),
        ("calculate 2 ** 8", "calculator", {"expression": "2 ** 8"}),
    ],
)
def test_first_decision_for_each_rule(prompt: str, tool: str, arguments: dict) -> None:
    assert decide(prompt) == ToolCallDecision(tool=tool, arguments=arguments)


def test_total_quantity_builds_a_sum_expression() -> None:
    history = (ok_step("search_records", {"records": [{"quantity": 3}, {"quantity": 4}]}),)
    decision = decide("What is the total quantity of records matching 'x'", history)
    assert decision == ToolCallDecision(tool="calculator", arguments={"expression": "3 + 4"})


def test_total_with_no_matches_answers_zero_without_calculator() -> None:
    history = (ok_step("search_records", {"records": []}),)
    decision = decide("What is the total value of records matching 'x'", history)
    assert decision == FinalAnswerDecision(answer="0")


def test_whole_number_float_is_answered_as_integer() -> None:
    history = (ok_step("lookup_record", {"record": {"unit_price": 32.0}}),)
    assert decide("What is the price of record D-21", history) == FinalAnswerDecision(answer="32")


def test_failed_observation_leads_to_explicit_inability_not_a_guess() -> None:
    failed = HistoryStep(
        step=1,
        decision=ToolCallDecision(tool="lookup_record", arguments={}),
        observation=Observation(
            call_id="call-1",
            tool="lookup_record",
            ok=False,
            error=ToolErrorInfo(type="ToolExecutionError", message="boom", retryable=False),
        ),
    )
    decision = decide("What is the quantity of record A-12", (failed,))
    assert decision == FinalAnswerDecision(answer="unable to answer: ToolExecutionError: boom")


def test_unmatched_prompt_raises() -> None:
    with pytest.raises(UnsupportedTaskError):
        decide("Tell me a joke")


def test_scripted_agent_replays_by_step_and_raises_when_exhausted() -> None:
    agent = ScriptedAgent([FinalAnswerDecision(answer="a")])
    assert asyncio.run(agent.decide(context("x"))) == FinalAnswerDecision(answer="a")
    with pytest.raises(ScriptExhaustedError):
        asyncio.run(agent.decide(context("x", (ok_step("calculator", {}),))))
