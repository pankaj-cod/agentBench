import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from agentbench.agents.rule_based import RuleBasedAgent
from agentbench.agents.scripted import ScriptedAgent
from agentbench.domain.benchmark import Expectations, NumericAnswer, TestCase, ToolExpectation
from agentbench.domain.evaluation import EvaluationOutcome, FailureCategory
from agentbench.domain.models import FinalAnswerDecision, RunResult, TokenUsage
from agentbench.evaluation.engine import EvaluationEngine
from agentbench.persistence.repositories import EvaluationConflictError
from agentbench.persistence.uow import UnitOfWork
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import default_registry
from agentbench.tracing.clock import FakeClock

from .test_persistence import insert_run

pytestmark = pytest.mark.postgres

CASE = TestCase(
    key="pct",
    prompt="What is 17% of the quantity of record A-12?",
    allowed_tools=("lookup_record", "calculator"),
    expectations=Expectations(
        final_answer=NumericAnswer(value=20.4, tolerance=0.01),
        required_tools=(ToolExpectation(name="lookup_record", args_subset={"record_id": "A-12"}),),
    ),
)


def execute(agent: Any = None) -> RunResult:
    runner = AgentRunner(default_registry(), clock=FakeClock())
    return asyncio.run(runner.run(CASE.to_task(), agent or RuleBasedAgent()))


def save_run(uow_factory: Callable[[], UnitOfWork], run: RunResult) -> None:
    with uow_factory() as uow:
        uow.agent_runs.add(run)
        uow.commit()


def test_evaluation_round_trips_with_outcome_category_explanation_and_version(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    run = execute(
        ScriptedAgent([FinalAnswerDecision(answer="17")])  # guesses, wrongly
    )
    evaluation = EvaluationEngine().evaluate(CASE, run)
    save_run(uow_factory, run)
    with uow_factory() as uow:
        assert uow.evaluations.save(evaluation) is True
        uow.commit()
    with uow_factory() as uow:
        loaded = uow.evaluations.get(run.run_id, evaluation.evaluator_version)
    assert loaded == evaluation
    assert loaded is not None
    assert loaded.outcome is EvaluationOutcome.FAILED
    assert loaded.failure_category is FailureCategory.MISSING_REQUIRED_TOOL
    assert "also failed: WRONG_ANSWER" in loaded.explanation


def test_rescoring_a_stored_trace_gives_the_same_evaluation(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    run = execute()
    original = EvaluationEngine().evaluate(CASE, run)
    save_run(uow_factory, run)
    with uow_factory() as uow:
        uow.evaluations.save(original)
        uow.commit()

    with uow_factory() as uow:
        reloaded_run = uow.agent_runs.get(run.run_id)
        assert reloaded_run is not None
        rescored = EvaluationEngine().evaluate(CASE, reloaded_run)
        assert rescored == original
        assert uow.evaluations.save(rescored) is False  # identical: nothing added
        uow.commit()
    with uow_factory() as uow:
        assert len(uow.evaluations.list_for_run(run.run_id)) == 1


def test_a_different_verdict_for_the_same_version_is_a_conflict(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    run = execute()
    evaluation = EvaluationEngine().evaluate(CASE, run)
    save_run(uow_factory, run)
    with uow_factory() as uow:
        uow.evaluations.save(evaluation)
        uow.commit()
    tampered = evaluation.model_copy(update={"explanation": "something else"})
    with uow_factory() as uow, pytest.raises(EvaluationConflictError):
        uow.evaluations.save(tampered)


def test_a_new_evaluator_version_adds_a_row_and_keeps_the_old_one(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    run = execute()
    v1 = EvaluationEngine().evaluate(CASE, run)
    v2 = v1.model_copy(update={"evaluator_version": "eval-2"})
    save_run(uow_factory, run)
    with uow_factory() as uow:
        assert uow.evaluations.save(v1) and uow.evaluations.save(v2)
        uow.commit()
    with uow_factory() as uow:
        versions = [e.evaluator_version for e in uow.evaluations.list_for_run(run.run_id)]
    assert versions == ["eval-1", "eval-2"]


def test_token_usage_round_trips_and_missing_usage_stays_null(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    usage = TokenUsage(model="example-model", input_tokens=40, output_tokens=4)
    with_usage = execute(ScriptedAgent([FinalAnswerDecision(answer="20.4", usage=usage)]))
    without = execute()
    save_run(uow_factory, with_usage)
    save_run(uow_factory, without)
    with uow_factory() as uow:
        a = uow.agent_runs.get(with_usage.run_id)
        b = uow.agent_runs.get(without.run_id)
    assert a is not None and (a.input_tokens, a.output_tokens) == (40, 4)
    assert b is not None and (b.input_tokens, b.output_tokens) == (None, None)


def test_deleting_a_run_deletes_its_evaluations(
    uow_factory: Callable[[], UnitOfWork], session_factory: sessionmaker[Session]
) -> None:
    run = execute()
    save_run(uow_factory, run)
    with uow_factory() as uow:
        uow.evaluations.save(EvaluationEngine().evaluate(CASE, run))
        uow.commit()
    with session_factory() as session:
        session.execute(text("DELETE FROM agent_run WHERE id = :id"), {"id": run.run_id})
        # Without commit, closing the session rolls the DELETE back (to the savepoint).
        session.commit()
    with uow_factory() as uow:
        assert uow.evaluations.list_for_run(run.run_id) == []


def insert_evaluation(session: Session, run_id: uuid.UUID, **overrides: Any) -> None:
    params = {
        "id": uuid.uuid4(),
        "agent_run_id": run_id,
        "evaluator_version": "eval-1",
        "pricing_version": None,
        "case_key": "k",
        "outcome": "PASSED",
        "failure_category": None,
        "failure_source": None,
        "explanation": "",
        "checks": "[]",
        "metrics": "{}",
        "expectations": "{}",
        **overrides,
    }
    json_columns = {"checks", "metrics", "expectations"}
    values = ", ".join(f"CAST(:{k} AS jsonb)" if k in json_columns else f":{k}" for k in params)
    session.execute(
        text(f"INSERT INTO evaluation_result ({', '.join(params)}) VALUES ({values})"), params
    )


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        ({"failure_category": "WRONG_ANSWER"}, "outcome_consistent"),  # PASSED with a category
        ({"outcome": "FAILED"}, "outcome_consistent"),  # FAILED without a category
        (
            {"outcome": "FAILED", "failure_category": "TOOL_FAILURE", "failure_source": "AGENT"},
            "category_matches_source",
        ),
        (
            {
                "outcome": "INVALID",
                "failure_category": "WRONG_ANSWER",
                "failure_source": "INFRASTRUCTURE",
            },
            "category_matches_source",
        ),
        ({"outcome": "MAYBE"}, "ck_evaluation_result_outcome"),
    ],
)
def test_evaluation_constraints(
    session_factory: sessionmaker[Session], overrides: dict[str, Any], constraint: str
) -> None:
    with session_factory() as session:
        run_id = insert_run(session)
        with pytest.raises(IntegrityError, match=constraint):
            insert_evaluation(session, run_id, **overrides)


def test_one_evaluation_per_run_and_evaluator_version(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = insert_run(session)
        insert_evaluation(session, run_id)
        with pytest.raises(IntegrityError, match="uq_evaluation_result_agent_run_id_evaluator"):
            insert_evaluation(session, run_id)


def test_valid_infrastructure_evaluation_is_accepted(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = insert_run(session)
        insert_evaluation(
            session,
            run_id,
            outcome="INVALID",
            failure_category="HARNESS_ERROR",
            failure_source="INFRASTRUCTURE",
            checks=json.dumps([]),
        )
        session.flush()
