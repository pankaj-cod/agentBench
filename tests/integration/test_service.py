import asyncio
from collections.abc import Callable

import pytest

from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.agents.scripted import ScriptedAgent
from agentbench.domain.enums import TerminationReason
from agentbench.domain.models import RunLimits, Task, ToolCallDecision
from agentbench.persistence.uow import UnitOfWork
from agentbench.runner.runner import AgentRunner
from agentbench.service import TaskRunService
from agentbench.tools.mock import default_registry
from agentbench.tracing.clock import FakeClock

pytestmark = pytest.mark.postgres

ALL_TOOLS = ("search_records", "lookup_record", "calculator")


def service(uow_factory: Callable[[], UnitOfWork]) -> TaskRunService:
    return TaskRunService(AgentRunner(default_registry(), clock=FakeClock()), uow_factory)


def test_run_and_save_persists_a_successful_run(uow_factory: Callable[[], UnitOfWork]) -> None:
    task = Task(
        key="total-value",
        prompt="What is the total value of records matching 'cable'?",
        allowed_tools=ALL_TOOLS,
    )
    result = asyncio.run(service(uow_factory).run_and_save(task, RuleBasedAgent()))

    with uow_factory() as uow:
        assert uow.agent_runs.get(result.run_id) == result
    assert result.final_answer == "468"


def test_run_and_save_persists_a_step_limit_failure(uow_factory: Callable[[], UnitOfWork]) -> None:
    task = Task(key="loop", prompt="search forever", allowed_tools=ALL_TOOLS)
    agent = ScriptedAgent([ToolCallDecision(tool="search_records", arguments={"query": "a"})] * 5)
    result = asyncio.run(service(uow_factory).run_and_save(task, agent, RunLimits(max_steps=2)))

    with uow_factory() as uow:
        saved = uow.agent_runs.get(result.run_id)
    assert saved is not None
    assert saved.termination_reason is TerminationReason.MAX_STEPS_EXCEEDED
    assert saved.events[-1].payload["termination_reason"] == "MAX_STEPS_EXCEEDED"
