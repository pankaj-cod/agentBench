import uuid
from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from agentbench.domain.benchmark import Expectations
from agentbench.domain.models import DomainModel


class EvaluationOutcome(StrEnum):
    PASSED = "PASSED"  # every check passed
    FAILED = "FAILED"  # the agent behaved incorrectly
    INVALID = "INVALID"  # infrastructure failed; the run says nothing about the agent


class FailureSource(StrEnum):
    AGENT = "AGENT"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class FailureCategory(StrEnum):
    """Listed in precedence order: when several checks fail, the first one listed here
    becomes the run's primary failure category. Execution problems come first because
    they make later checks meaningless."""

    HARNESS_ERROR = "HARNESS_ERROR"
    TOOL_FAILURE = "TOOL_FAILURE"
    AGENT_ERROR = "AGENT_ERROR"
    TIMEOUT = "TIMEOUT"
    STEP_LIMIT = "STEP_LIMIT"
    FORBIDDEN_TOOL_USED = "FORBIDDEN_TOOL_USED"
    INVALID_TOOL_CALL = "INVALID_TOOL_CALL"
    TOOL_CALL_LIMIT_EXCEEDED = "TOOL_CALL_LIMIT_EXCEEDED"
    MISSING_REQUIRED_TOOL = "MISSING_REQUIRED_TOOL"
    WRONG_TOOL_ARGUMENTS = "WRONG_TOOL_ARGUMENTS"
    MISSING_RETRIEVED_RECORDS = "MISSING_RETRIEVED_RECORDS"
    WRONG_ANSWER = "WRONG_ANSWER"

    @property
    def source(self) -> FailureSource:
        if self in (FailureCategory.HARNESS_ERROR, FailureCategory.TOOL_FAILURE):
            return FailureSource.INFRASTRUCTURE
        return FailureSource.AGENT

    @property
    def precedence(self) -> int:
        return list(FailureCategory).index(self)


class CheckResult(DomainModel):
    name: str
    passed: bool
    failure_category: FailureCategory | None
    explanation: str

    @model_validator(mode="after")
    def _category_iff_failed(self) -> Self:
        if self.passed != (self.failure_category is None):
            raise ValueError("a failed check needs a category; a passed check has none")
        return self


class RunMetrics(DomainModel):
    """Per-run measurements taken from the trace. None means "not measurable for this
    run", never zero."""

    steps: int = Field(ge=0)
    duration_us: int = Field(ge=0)
    tool_calls: int = Field(ge=0)  # tool-call decisions
    tool_calls_unknown_or_not_allowed: int = Field(ge=0)  # rejected before argument checks
    tool_calls_invalid_arguments: int = Field(ge=0)  # rejected by argument validation
    tool_attempts: int = Field(ge=0)  # executions started, including retries
    tool_attempts_failed: int = Field(ge=0)
    tool_selection_correct: bool | None  # None when the case has no tool expectations
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    estimated_cost_usd: Decimal | None = Field(ge=0)


class Evaluation(DomainModel):
    run_id: uuid.UUID
    case_key: str
    evaluator_version: str
    pricing_version: str | None
    outcome: EvaluationOutcome
    failure_category: FailureCategory | None
    failure_source: FailureSource | None
    explanation: str
    checks: tuple[CheckResult, ...]
    metrics: RunMetrics
    expectations: Expectations  # the exact criteria this verdict was based on

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        category, source = self.failure_category, self.failure_source
        if self.outcome is EvaluationOutcome.PASSED:
            ok = category is None and source is None
        elif self.outcome is EvaluationOutcome.FAILED:
            ok = category is not None and source is FailureSource.AGENT
        else:
            ok = category is not None and source is FailureSource.INFRASTRUCTURE
        if not ok:
            raise ValueError(
                f"outcome {self.outcome} is inconsistent with category {category} "
                f"and source {source}"
            )
        if category is not None and category.source is not source:
            raise ValueError(f"category {category} belongs to source {category.source}")
        return self
