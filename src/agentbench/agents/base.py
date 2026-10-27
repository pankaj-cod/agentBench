from typing import Any, Protocol

from agentbench.domain.models import (
    DomainModel,
    FinalAnswerDecision,
    HistoryStep,
    Task,
    ToolCallDecision,
)


class AgentContext(DomainModel):
    """Everything an agent may use to make its next decision."""

    task: Task
    step: int  # 1-based number of the decision being requested
    tools: tuple[dict[str, Any], ...]  # schemas of the tools this task allows
    history: tuple[HistoryStep, ...]  # earlier decisions and what they returned


class Agent(Protocol):
    name: str
    version: str

    async def decide(self, context: AgentContext) -> ToolCallDecision | FinalAnswerDecision: ...
