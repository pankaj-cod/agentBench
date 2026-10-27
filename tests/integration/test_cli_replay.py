import io
import json
from pathlib import Path

import pytest
from sqlalchemy import Engine

from agentbench.cli import main
from agentbench.domain.benchmark import Expectations, NumericAnswer, TestCase, ToolExpectation
from agentbench.evaluation.engine import EvaluationEngine
from agentbench.persistence.db import DATABASE_URL_ENV, create_session_factory
from agentbench.persistence.uow import UnitOfWork

from .test_versioning_persistence import model_run

pytestmark = pytest.mark.postgres

CASE = TestCase(
    key="q",
    prompt="quantity of A-12?",
    allowed_tools=("lookup_record",),
    expectations=Expectations(
        final_answer=NumericAnswer(value=120),
        required_tools=(ToolExpectation(name="lookup_record"),),
    ),
)


@pytest.fixture
def saved_run_id(migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> str:
    # The CLI opens its own connection, so this is committed to the throwaway session
    # database (dropped when the test session ends).
    monkeypatch.setenv(DATABASE_URL_ENV, migrated_engine.url.render_as_string(hide_password=False))
    run = model_run()
    with UnitOfWork(create_session_factory(migrated_engine)) as uow:
        uow.agent_runs.add(run)
        uow.evaluations.save(EvaluationEngine().evaluate(CASE, run))
        uow.commit()
    return str(run.run_id)


def replay(*args: str) -> tuple[int, dict]:
    out = io.StringIO()
    code = main(["replay", *args], out=out)
    return code, json.loads(out.getvalue()) if out.getvalue() else {}


def test_trace_replay_from_the_command_line(saved_run_id: str) -> None:
    code, result = replay(saved_run_id, "--mode", "trace")
    assert code == 0
    assert result["mode"] == "TRACE" and result["deterministic"] is True
    assert result["new_run_id"] is None
    assert result["evaluation"]["outcome"] == "PASSED"


def test_recorded_reexecution_is_saved_with_lineage(saved_run_id: str) -> None:
    code, result = replay(saved_run_id, "--mode", "recorded", "--save")
    assert code == 0
    assert result["mode"] == "RECORDED_RESPONSES" and result["deterministic"] is True
    out = io.StringIO()
    assert main(["show", result["new_run_id"], "--json"], out=out) == 0
    new_run = json.loads(out.getvalue())
    assert new_run["source_run_id"] == saved_run_id
    assert new_run["execution_mode"] == "RECORDED_RESPONSES"


def test_live_replay_of_a_scripted_run_is_refused(saved_run_id: str) -> None:
    code, _ = replay(saved_run_id, "--mode", "live")
    assert code == 1


def test_bench_with_an_unusable_agent_config_fails_cleanly(tmp_path: Path) -> None:
    config = tmp_path / "scripted.json"
    config.write_text(
        json.dumps(
            {
                "label": "x",
                "provider": "scripted",
                "model": "m",
                "prompt": {"id": "p", "version": "1", "system": "s"},
            }
        )
    )
    example = Path(__file__).resolve().parents[2] / "examples/benchmarks/inventory-v1.json"
    out = io.StringIO()
    assert (
        main(["bench", str(example), "--label", "x", "--agent-config", str(config)], out=out) == 1
    )
