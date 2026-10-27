from typing import Any

from agentbench.domain.models import ToolErrorInfo


class AgentBenchError(Exception):
    """Base class for all errors raised by AgentBench."""


class InvalidToolDefinitionError(AgentBenchError):
    """A tool cannot be registered: bad name, duplicate, or unsafe argument schema."""


class InvalidTaskError(AgentBenchError):
    """A task is misconfigured, e.g. it allows a tool that is not registered."""


class ToolError(AgentBenchError):
    """A failure while validating or executing one tool call. Recorded in the trace and
    shown to the agent; never raised out of AgentRunner.run."""

    def __init__(
        self, message: str, *, retryable: bool = False, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.details = details or {}

    def to_info(self) -> ToolErrorInfo:
        return ToolErrorInfo(
            type=type(self).__name__,
            message=self.message,
            retryable=self.retryable,
            details=self.details,
        )


class ToolValidationError(ToolError):
    """The call was rejected before the tool ran. Never retried."""


class UnknownToolError(ToolValidationError):
    pass


class ToolNotAllowedError(ToolValidationError):
    pass


class ToolArgumentsError(ToolValidationError):
    pass


class ToolTimeoutError(ToolError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, retryable=True, details=details)


class ToolExecutionError(ToolError):
    """The tool ran and failed. `retryable` says whether trying again could help."""
