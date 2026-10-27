from collections.abc import Callable

from agentbench.agents.base import Agent
from agentbench.domain.models import RunLimits, RunResult, Task
from agentbench.persistence.uow import UnitOfWork
from agentbench.runner.runner import AgentRunner


class TaskRunService:
    """Runs one task and saves the run and its full trace.

    The agent runs with no database transaction open. The result is then saved in a
    single transaction: the run and all its events are stored together or not at all.
    """

    def __init__(self, runner: AgentRunner, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._runner = runner
        self._uow_factory = uow_factory

    async def run_and_save(
        self, task: Task, agent: Agent, limits: RunLimits | None = None
    ) -> RunResult:
        result = await self._runner.run(task, agent, limits)
        with self._uow_factory() as uow:
            uow.agent_runs.add(result)
            uow.commit()
        return result
