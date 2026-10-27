"""Migration 0003: versioning and lineage columns, and the four CHECK constraints that
were changed by hand to allow MODEL_UNAVAILABLE."""

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from agentbench.agents.model_agent import ModelAgent
from agentbench.domain.agent_config import ModelAgentConfig, PromptTemplate
from agentbench.domain.enums import ExecutionMode
from agentbench.domain.models import RunResult, Task
from agentbench.llm.base import ModelResponse, ModelToolCall
from agentbench.llm.scripted import ScriptedModelAdapter
from agentbench.persistence.uow import UnitOfWork
from agentbench.runner.runner import AgentRunner
from agentbench.tools.mock import default_registry
from agentbench.tracing.clock import FakeClock

from .test_evaluation_persistence import insert_evaluation
from .test_persistence import insert_run

pytestmark = pytest.mark.postgres

CONFIG = ModelAgentConfig(
    label="t",
    provider="scripted",
    model="fake-model-1",
    prompt=PromptTemplate(id="p", version="1", system="s"),
)


def model_run(**run_kwargs: Any) -> RunResult:
    adapter = ScriptedModelAdapter(
        [
            ModelResponse(
                text=None,
                tool_calls=(
                    ModelToolCall(id="t1", name="lookup_record", arguments={"record_id": "A-12"}),
                ),
                stop_reason="tool_use",
                usage=None,
            ),
            ModelResponse(text="120", tool_calls=(), stop_reason="end_turn", usage=None),
        ]
    )
    task = Task(key="q", prompt="quantity of A-12?", allowed_tools=("lookup_record",))
    agent = ModelAgent(CONFIG, adapter, clock=FakeClock())
    runner = AgentRunner(default_registry(), clock=FakeClock())
    return asyncio.run(runner.run(task, agent, **run_kwargs))


def save(uow_factory: Callable[[], UnitOfWork], result: RunResult) -> None:
    with uow_factory() as uow:
        uow.agent_runs.add(result)
        uow.commit()


def test_model_run_round_trips_with_versions_and_lineage(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    source = model_run()
    rerun = model_run(execution_mode=ExecutionMode.RECORDED_RESPONSES, source_run_id=source.run_id)
    save(uow_factory, source)
    save(uow_factory, rerun)
    with uow_factory() as uow:
        loaded = uow.agent_runs.get(rerun.run_id)
    assert loaded == rerun
    assert loaded is not None
    assert loaded.agent_config_hash == CONFIG.config_hash
    assert loaded.tool_schemas_hash == source.tool_schemas_hash
    assert loaded.execution_mode is ExecutionMode.RECORDED_RESPONSES
    assert loaded.source_run_id == source.run_id


def test_deleting_the_source_run_keeps_the_rerun_and_clears_the_link(
    uow_factory: Callable[[], UnitOfWork], session_factory: sessionmaker[Session]
) -> None:
    source = model_run()
    rerun = model_run(source_run_id=source.run_id)
    save(uow_factory, source)
    save(uow_factory, rerun)
    with session_factory() as session:
        session.execute(text("DELETE FROM agent_run WHERE id = :id"), {"id": source.run_id})
        session.commit()
    with uow_factory() as uow:
        loaded = uow.agent_runs.get(rerun.run_id)
    assert loaded is not None and loaded.source_run_id is None


def test_model_unavailable_run_and_invalid_evaluation_are_accepted(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = insert_run(
            session,
            status="FAILED",
            termination_reason="MODEL_UNAVAILABLE",
            final_answer=None,
        )
        insert_evaluation(
            session,
            run_id,
            outcome="INVALID",
            failure_category="MODEL_UNAVAILABLE",
            failure_source="INFRASTRUCTURE",
        )
        session.flush()


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        (
            {"status": "ERROR", "termination_reason": "MODEL_UNAVAILABLE", "final_answer": None},
            "ck_agent_run_status_matches_reason",
        ),
        ({"execution_mode": "SOMETIMES"}, "ck_agent_run_execution_mode"),
        ({"source_run_id": uuid.uuid4()}, "fk_agent_run_source_run_id_agent_run"),
    ],
)
def test_versioning_constraints_on_agent_run(
    session_factory: sessionmaker[Session], overrides: dict[str, Any], constraint: str
) -> None:
    with session_factory() as session, pytest.raises(IntegrityError, match=constraint):
        insert_run(session, **overrides)


def test_model_unavailable_cannot_be_blamed_on_the_agent(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = insert_run(
            session, status="FAILED", termination_reason="MODEL_UNAVAILABLE", final_answer=None
        )
        with pytest.raises(IntegrityError, match="category_matches_source"):
            insert_evaluation(
                session,
                run_id,
                outcome="FAILED",
                failure_category="MODEL_UNAVAILABLE",
                failure_source="AGENT",
            )
