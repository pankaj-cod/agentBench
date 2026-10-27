import uuid

import pytest
from pydantic import ValidationError

from agentbench.domain.benchmark import (
    Benchmark,
    Expectations,
    NumericAnswer,
    TestCase,
    ToolExpectation,
)
from agentbench.domain.evaluation import (
    CheckResult,
    Evaluation,
    EvaluationOutcome,
    FailureCategory,
    FailureSource,
    RunMetrics,
)


def make_case(key: str = "a", value: float = 1.0) -> TestCase:
    return TestCase(
        key=key,
        prompt="p",
        expectations=Expectations(final_answer=NumericAnswer(value=value)),
    )


def test_empty_benchmark_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        Benchmark(name="b", version="1", cases=())


def test_duplicate_case_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Benchmark(name="b", version="1", cases=(make_case("a"), make_case("a")))


def test_tool_both_required_and_forbidden_is_rejected() -> None:
    with pytest.raises(ValidationError, match="required and forbidden"):
        Expectations(
            required_tools=(ToolExpectation(name="calculator"),), forbidden_tools=("calculator",)
        )


def test_content_hash_changes_with_content_and_only_with_content() -> None:
    first = Benchmark(name="b", version="1", cases=(make_case(value=1.0),))
    same = Benchmark(name="b", version="1", cases=(make_case(value=1.0),))
    changed = Benchmark(name="b", version="1", cases=(make_case(value=2.0),))
    assert first.content_hash == same.content_hash
    assert first.content_hash != changed.content_hash


METRICS = RunMetrics(
    steps=1,
    duration_us=1,
    tool_calls=0,
    tool_calls_unknown_or_not_allowed=0,
    tool_calls_invalid_arguments=0,
    tool_attempts=0,
    tool_attempts_failed=0,
    tool_selection_correct=None,
    input_tokens=None,
    output_tokens=None,
    estimated_cost_usd=None,
)


@pytest.mark.parametrize(
    ("outcome", "category", "source"),
    [
        (EvaluationOutcome.PASSED, FailureCategory.WRONG_ANSWER, FailureSource.AGENT),
        (EvaluationOutcome.FAILED, None, None),
        (EvaluationOutcome.FAILED, FailureCategory.TOOL_FAILURE, FailureSource.INFRASTRUCTURE),
        (EvaluationOutcome.INVALID, FailureCategory.WRONG_ANSWER, FailureSource.AGENT),
        (EvaluationOutcome.FAILED, FailureCategory.HARNESS_ERROR, FailureSource.AGENT),
    ],
)
def test_evaluation_rejects_inconsistent_outcome_category_and_source(
    outcome: EvaluationOutcome, category: FailureCategory | None, source: FailureSource | None
) -> None:
    with pytest.raises(ValidationError):
        Evaluation(
            run_id=uuid.uuid4(),
            case_key="a",
            evaluator_version="eval-1",
            pricing_version=None,
            outcome=outcome,
            failure_category=category,
            failure_source=source,
            explanation="x",
            checks=(),
            metrics=METRICS,
            expectations=Expectations(),
        )


def test_check_result_needs_category_exactly_when_failed() -> None:
    with pytest.raises(ValidationError):
        CheckResult(name="x", passed=False, failure_category=None, explanation="")


def test_only_harness_tool_and_model_availability_failures_are_infrastructure() -> None:
    infra = {c for c in FailureCategory if c.source is FailureSource.INFRASTRUCTURE}
    assert infra == {
        FailureCategory.HARNESS_ERROR,
        FailureCategory.TOOL_FAILURE,
        FailureCategory.MODEL_UNAVAILABLE,
    }
