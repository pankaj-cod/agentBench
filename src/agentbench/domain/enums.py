from enum import StrEnum


class AgentRunStatus(StrEnum):
    """How execution ended. Says nothing about whether the answer was correct."""

    COMPLETED = "COMPLETED"  # the agent gave a final answer
    FAILED = "FAILED"  # the agent or its tools caused the run to stop
    ERROR = "ERROR"  # AgentBench itself failed; the result says nothing about the agent


class TerminationReason(StrEnum):
    COMPLETED = "COMPLETED"
    MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"
    RUN_TIMEOUT = "RUN_TIMEOUT"
    TOOL_FAILURE_EXHAUSTED = "TOOL_FAILURE_EXHAUSTED"
    AGENT_ERROR = "AGENT_ERROR"
    HARNESS_ERROR = "HARNESS_ERROR"

    @property
    def status(self) -> AgentRunStatus:
        if self is TerminationReason.COMPLETED:
            return AgentRunStatus.COMPLETED
        if self is TerminationReason.HARNESS_ERROR:
            return AgentRunStatus.ERROR
        return AgentRunStatus.FAILED


class TraceEventType(StrEnum):
    RUN_STARTED = "RUN_STARTED"
    AGENT_DECISION = "AGENT_DECISION"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_SUCCEEDED = "TOOL_CALL_SUCCEEDED"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    FINAL_ANSWER = "FINAL_ANSWER"
    RUN_FINISHED = "RUN_FINISHED"
