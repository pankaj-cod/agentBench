import hashlib
import json
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from agentbench.domain.models import DomainModel, Slug, Task

ToolName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class ExactAnswer(DomainModel):
    """The whole answer, with surrounding whitespace removed, must equal `value`."""

    type: Literal["exact"] = "exact"
    value: str
    case_sensitive: bool = False


class NumericAnswer(DomainModel):
    """The whole answer must be a finite number within `tolerance` of `value`."""

    type: Literal["numeric"] = "numeric"
    value: float
    tolerance: float = Field(default=0.0, ge=0)


class RegexAnswer(DomainModel):
    """The whole answer must match `pattern` (Python `re.fullmatch`)."""

    type: Literal["regex"] = "regex"
    pattern: str = Field(min_length=1, max_length=500)


class OneOfAnswer(DomainModel):
    """The answer must equal one of `values` (case-insensitive, whitespace stripped)."""

    type: Literal["one_of"] = "one_of"
    values: tuple[str, ...] = Field(min_length=1)


AnswerExpectation = Annotated[
    ExactAnswer | NumericAnswer | RegexAnswer | OneOfAnswer, Field(discriminator="type")
]


class ToolExpectation(DomainModel):
    """A tool that must be called *successfully* at least once. If `args_subset` is not
    empty, at least one successful call must have those argument values."""

    name: ToolName
    args_subset: dict[str, Any] = Field(default_factory=dict)


class Expectations(DomainModel):
    final_answer: AnswerExpectation | None = None
    required_tools: tuple[ToolExpectation, ...] = ()
    forbidden_tools: tuple[ToolName, ...] = ()
    max_tool_calls: int | None = Field(default=None, ge=0)
    # Record ids that must appear in successful search_records / lookup_record outputs.
    expected_records: tuple[str, ...] = ()
    allow_invalid_tool_calls: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        overlap = {t.name for t in self.required_tools} & set(self.forbidden_tools)
        if overlap:
            raise ValueError(f"tools both required and forbidden: {sorted(overlap)}")
        return self

    @property
    def has_tool_expectations(self) -> bool:
        return bool(self.required_tools or self.forbidden_tools)


class TestCase(DomainModel):
    __test__ = False  # not a pytest test class

    key: Slug
    prompt: str = Field(min_length=1, max_length=20_000)
    allowed_tools: tuple[ToolName, ...] = ()
    context: dict[str, Any] = Field(default_factory=dict)
    expectations: Expectations

    def to_task(self) -> Task:
        return Task(
            key=self.key, prompt=self.prompt, allowed_tools=self.allowed_tools, context=self.context
        )


class Benchmark(DomainModel):
    name: Slug
    version: str = Field(pattern=r"^[A-Za-z0-9._-]{1,50}$")
    description: str = Field(default="", max_length=2_000)
    cases: tuple[TestCase, ...] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _unique_keys(self) -> Self:
        keys = [c.key for c in self.cases]
        if len(keys) != len(set(keys)):
            raise ValueError("test case keys must be unique")
        return self

    @property
    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON form; two benchmarks compare as the same only
        if their content is identical."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class BenchmarkRef(DomainModel):
    name: str
    version: str
    content_hash: str = Field(min_length=64, max_length=64)

    @classmethod
    def of(cls, benchmark: Benchmark) -> "BenchmarkRef":
        return cls(
            name=benchmark.name, version=benchmark.version, content_hash=benchmark.content_hash
        )
