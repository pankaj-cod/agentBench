import uuid
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from agentbench.domain.enums import AgentRunStatus, TerminationReason, TraceEventType


class DomainModel(BaseModel):
    """Base for all domain models: unknown fields are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,99}$")]


class RunLimits(DomainModel):
    max_steps: int = Field(default=10, ge=1, le=100)
    tool_timeout_s: float = Field(default=5.0, gt=0, le=60)
    run_timeout_s: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_s: float = Field(default=0.1, ge=0, le=10)


class Task(DomainModel):
    key: Slug
    prompt: str = Field(min_length=1, max_length=20_000)
    allowed_tools: tuple[str, ...] = ()
    context: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(DomainModel):
    """Model usage reported by an agent for one decision. Agents that use no model
    report nothing (None), which is different from reporting zero tokens."""

    model: str = Field(min_length=1, max_length=100)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ToolCallDecision(DomainModel):
    kind: Literal["tool_call"] = "tool_call"
    tool: str
    # Untrusted: whatever the agent produced. The ToolRegistry validates it.
    arguments: Any = Field(default_factory=dict)
    usage: TokenUsage | None = None


class FinalAnswerDecision(DomainModel):
    kind: Literal["final_answer"] = "final_answer"
    answer: str
    usage: TokenUsage | None = None


Decision = Annotated[ToolCallDecision | FinalAnswerDecision, Field(discriminator="kind")]


class ToolErrorInfo(DomainModel):
    type: str
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class Observation(DomainModel):
    """What the agent learns after a tool call: an output or an error, never both."""

    call_id: str
    tool: str
    ok: bool
    output: Any = None
    error: ToolErrorInfo | None = None

    @model_validator(mode="after")
    def _output_xor_error(self) -> Self:
        if self.ok and self.error is not None:
            raise ValueError("a successful observation cannot carry an error")
        if not self.ok and self.error is None:
            raise ValueError("a failed observation must carry an error")
        return self


class HistoryStep(DomainModel):
    step: int = Field(ge=1)
    decision: Decision
    observation: Observation | None = None


class TraceEvent(DomainModel):
    id: uuid.UUID
    run_id: uuid.UUID
    seq: int = Field(ge=0)
    step: int = Field(ge=0)
    type: TraceEventType
    occurred_at: AwareDatetime
    elapsed_us: int = Field(ge=0)
    payload: dict[str, Any]


class RunResult(DomainModel):
    run_id: uuid.UUID
    task: Task
    limits: RunLimits
    agent_name: str
    agent_version: str
    status: AgentRunStatus
    termination_reason: TerminationReason
    final_answer: str | None
    error_message: str | None
    step_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    started_at: AwareDatetime
    finished_at: AwareDatetime
    duration_us: int = Field(ge=0)
    # Totals over all decisions, or None unless every decision reported usage.
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    events: tuple[TraceEvent, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.termination_reason.status is not self.status:
            raise ValueError(
                f"status {self.status} does not match termination reason {self.termination_reason}"
            )
        if self.status is AgentRunStatus.COMPLETED and self.final_answer is None:
            raise ValueError("a COMPLETED run must have a final answer")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at is before started_at")
        if [e.seq for e in self.events] != list(range(len(self.events))):
            raise ValueError("event seq values must be 0..n-1 in order")
        if any(e.run_id != self.run_id for e in self.events):
            raise ValueError("every event must belong to this run")
        return self
