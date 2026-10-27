from collections.abc import Sequence

from agentbench.agents.base import AgentContext
from agentbench.domain.errors import AgentBenchError
from agentbench.domain.models import FinalAnswerDecision, ToolCallDecision


class ScriptExhaustedError(AgentBenchError):
    pass


class ScriptedAgent:
    """Returns the decision at position `step - 1` of a fixed script, ignoring what the
    tools return. Used to reproduce exact behaviour, including deliberately wrong calls
    such as unknown tools or malformed arguments."""

    name = "scripted"
    version = "1"

    def __init__(self, decisions: Sequence[ToolCallDecision | FinalAnswerDecision]) -> None:
        self._decisions = tuple(decisions)

    async def decide(self, context: AgentContext) -> ToolCallDecision | FinalAnswerDecision:
        index = context.step - 1
        if index >= len(self._decisions):
            raise ScriptExhaustedError(
                f"script has {len(self._decisions)} decisions, step {context.step} requested"
            )
        return self._decisions[index]
