"""Deterministic evaluation of one run against one test case.

`EvaluationEngine.evaluate` is a pure function of (test case, run result): no clocks,
no randomness, no network, no LLM. Evaluating the same trace twice gives equal
results, which is what makes stored runs re-scorable.
"""

import math
import re

from agentbench.domain.benchmark import (
    AnswerExpectation,
    ExactAnswer,
    Expectations,
    NumericAnswer,
    OneOfAnswer,
    RegexAnswer,
    TestCase,
    ToolExpectation,
)
from agentbench.domain.enums import TerminationReason
from agentbench.domain.evaluation import (
    CheckResult,
    Evaluation,
    EvaluationOutcome,
    FailureCategory,
    FailureSource,
    RunMetrics,
)
from agentbench.domain.models import RunResult
from agentbench.evaluation.pricing import EMPTY_PRICE_TABLE, PriceTable
from agentbench.evaluation.trace_view import TraceView

EVALUATOR_VERSION = "eval-1"  # bump whenever a check changes meaning

TERMINATION_CATEGORY: dict[TerminationReason, FailureCategory] = {
    TerminationReason.HARNESS_ERROR: FailureCategory.HARNESS_ERROR,
    TerminationReason.TOOL_FAILURE_EXHAUSTED: FailureCategory.TOOL_FAILURE,
    TerminationReason.AGENT_ERROR: FailureCategory.AGENT_ERROR,
    TerminationReason.RUN_TIMEOUT: FailureCategory.TIMEOUT,
    TerminationReason.MAX_STEPS_EXCEEDED: FailureCategory.STEP_LIMIT,
}


def _precedence(check: CheckResult) -> int:
    assert check.failure_category is not None
    return check.failure_category.precedence


def _passed(name: str, explanation: str) -> CheckResult:
    return CheckResult(name=name, passed=True, failure_category=None, explanation=explanation)


def _failed(name: str, category: FailureCategory, explanation: str) -> CheckResult:
    return CheckResult(name=name, passed=False, failure_category=category, explanation=explanation)


class EvaluationEngine:
    def __init__(self, prices: PriceTable = EMPTY_PRICE_TABLE) -> None:
        self.version = EVALUATOR_VERSION
        self._prices = prices

    def evaluate(self, case: TestCase, run: RunResult) -> Evaluation:
        if run.task.key != case.key:
            raise ValueError(f"run is for task {run.task.key!r}, not case {case.key!r}")
        view = TraceView.from_events(run.events)
        expected = case.expectations
        checks = [self._execution(run)]
        if expected.final_answer is not None:
            checks.append(self._final_answer(expected.final_answer, run.final_answer))
        checks += [self._required_tool(t, view) for t in expected.required_tools]
        if expected.forbidden_tools:
            checks.append(self._forbidden_tools(expected.forbidden_tools, view))
        if not expected.allow_invalid_tool_calls:
            checks.append(self._valid_tool_calls(view))
        if expected.max_tool_calls is not None:
            checks.append(self._max_tool_calls(expected.max_tool_calls, view))
        if expected.expected_records:
            checks.append(self._retrieved_records(expected.expected_records, view))

        failing = [c for c in checks if not c.passed]
        if not failing:
            outcome, category, source = EvaluationOutcome.PASSED, None, None
            explanation = f"all {len(checks)} checks passed"
        else:
            primary = min(failing, key=_precedence)
            category = primary.failure_category
            assert category is not None
            source = category.source
            outcome = (
                EvaluationOutcome.FAILED
                if source is FailureSource.AGENT
                else EvaluationOutcome.INVALID
            )
            explanation = f"{category}: {primary.explanation}"
            if len(failing) > 1:
                others = ", ".join(str(c.failure_category) for c in failing if c is not primary)
                explanation += f" (also failed: {others})"

        return Evaluation(
            run_id=run.run_id,
            case_key=case.key,
            evaluator_version=self.version,
            pricing_version=self._prices.version,
            outcome=outcome,
            failure_category=category,
            failure_source=source,
            explanation=explanation,
            checks=tuple(checks),
            metrics=self._metrics(run, view, expected),
            expectations=expected,
        )

    # --- checks -------------------------------------------------------------------

    @staticmethod
    def _execution(run: RunResult) -> CheckResult:
        if run.termination_reason is TerminationReason.COMPLETED:
            return _passed("execution", "the agent returned a final answer")
        detail = f"run ended with {run.termination_reason}"
        if run.error_message:
            detail += f": {run.error_message}"
        return _failed("execution", TERMINATION_CATEGORY[run.termination_reason], detail)

    @staticmethod
    def _final_answer(expected: AnswerExpectation, answer: str | None) -> CheckResult:
        name = "final_answer"
        if answer is None:
            return _failed(name, FailureCategory.WRONG_ANSWER, "no final answer")
        got = answer.strip()
        match expected:
            case NumericAnswer(value=value, tolerance=tol):
                try:
                    number = float(got)
                except ValueError:
                    return _failed(
                        name, FailureCategory.WRONG_ANSWER, f"expected a number, got {got!r}"
                    )
                if not math.isfinite(number):
                    return _failed(name, FailureCategory.WRONG_ANSWER, f"not finite: {got!r}")
                if abs(number - value) <= tol:
                    return _passed(name, f"{number:g} is within {tol:g} of {value:g}")
                return _failed(
                    name, FailureCategory.WRONG_ANSWER, f"expected {value:g} ± {tol:g}, got {got!r}"
                )
            case ExactAnswer(value=value, case_sensitive=cs):
                ok = got == value.strip() if cs else got.casefold() == value.strip().casefold()
                if ok:
                    return _passed(name, f"answer equals {value!r}")
                return _failed(
                    name, FailureCategory.WRONG_ANSWER, f"expected {value!r}, got {got!r}"
                )
            case RegexAnswer(pattern=pattern):
                if re.fullmatch(pattern, got):
                    return _passed(name, f"answer matches /{pattern}/")
                return _failed(
                    name, FailureCategory.WRONG_ANSWER, f"{got!r} does not match /{pattern}/"
                )
            case OneOfAnswer(values=values):
                if got.casefold() in {v.strip().casefold() for v in values}:
                    return _passed(name, "answer is one of the accepted values")
                return _failed(
                    name,
                    FailureCategory.WRONG_ANSWER,
                    f"expected one of {list(values)}, got {got!r}",
                )
        raise AssertionError(f"unhandled answer expectation: {expected!r}")

    @staticmethod
    def _required_tool(expected: ToolExpectation, view: TraceView) -> CheckResult:
        name = f"required_tool:{expected.name}"
        successes = [c for c in view.calls if c.tool == expected.name and c.succeeded]
        if not successes:
            attempted = any(c.tool == expected.name for c in view.calls)
            reason = "was requested but never succeeded" if attempted else "was never called"
            return _failed(name, FailureCategory.MISSING_REQUIRED_TOOL, f"{expected.name} {reason}")
        for call in successes:
            args = call.validated_arguments or {}
            if all(k in args and args[k] == v for k, v in expected.args_subset.items()):
                return _passed(name, f"{expected.name} succeeded with the expected arguments")
        seen = [c.validated_arguments for c in successes]
        return _failed(
            name,
            FailureCategory.WRONG_TOOL_ARGUMENTS,
            f"{expected.name} expected arguments including {expected.args_subset}, got {seen}",
        )

    @staticmethod
    def _forbidden_tools(forbidden: tuple[str, ...], view: TraceView) -> CheckResult:
        used = sorted({c.tool for c in view.calls if c.tool in forbidden})
        if used:
            return _failed(
                "forbidden_tools", FailureCategory.FORBIDDEN_TOOL_USED, f"requested {used}"
            )
        return _passed("forbidden_tools", "no forbidden tool was requested")

    @staticmethod
    def _valid_tool_calls(view: TraceView) -> CheckResult:
        invalid = [c for c in view.calls if c.validation_error is not None]
        if invalid:
            detail = ", ".join(f"{c.call_id} {c.tool}: {c.validation_error}" for c in invalid)
            return _failed("valid_tool_calls", FailureCategory.INVALID_TOOL_CALL, detail)
        return _passed("valid_tool_calls", f"all {len(view.calls)} tool calls passed validation")

    @staticmethod
    def _max_tool_calls(limit: int, view: TraceView) -> CheckResult:
        if len(view.calls) > limit:
            return _failed(
                "max_tool_calls",
                FailureCategory.TOOL_CALL_LIMIT_EXCEEDED,
                f"{len(view.calls)} tool calls, limit {limit}",
            )
        return _passed("max_tool_calls", f"{len(view.calls)} tool calls, limit {limit}")

    @staticmethod
    def _retrieved_records(expected: tuple[str, ...], view: TraceView) -> CheckResult:
        missing = sorted(set(expected) - view.retrieved_record_ids())
        if missing:
            return _failed(
                "retrieved_records",
                FailureCategory.MISSING_RETRIEVED_RECORDS,
                f"never retrieved {missing}",
            )
        return _passed("retrieved_records", f"retrieved all of {sorted(expected)}")

    # --- metrics ------------------------------------------------------------------

    def _metrics(self, run: RunResult, view: TraceView, expected: Expectations) -> RunMetrics:
        selection_correct = None
        if expected.has_tool_expectations:
            called = {c.tool for c in view.calls}
            selection_correct = (
                all(t.name in called for t in expected.required_tools)
                and not called & set(expected.forbidden_tools)
                and not any(c.selection_error for c in view.calls)
            )
        return RunMetrics(
            steps=run.step_count,
            duration_us=run.duration_us,
            tool_calls=len(view.calls),
            tool_calls_unknown_or_not_allowed=sum(c.selection_error for c in view.calls),
            tool_calls_invalid_arguments=sum(c.argument_error for c in view.calls),
            tool_attempts=view.tool_attempts,
            tool_attempts_failed=view.failed_tool_attempts,
            tool_selection_correct=selection_correct,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            estimated_cost_usd=self._prices.cost(view.usages) if view.usages else None,
        )
