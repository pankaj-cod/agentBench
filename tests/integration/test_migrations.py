import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import URL, create_engine, inspect

from agentbench.persistence.orm import Base

from .conftest import alembic_config

pytestmark = pytest.mark.postgres


def table_names(url: URL) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_upgrade_creates_a_schema_identical_to_the_orm(empty_database: URL) -> None:
    command.upgrade(alembic_config(empty_database), "head")
    engine = create_engine(empty_database)
    try:
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    finally:
        engine.dispose()
    assert diff == []


def test_downgrade_removes_everything_and_upgrade_works_again(empty_database: URL) -> None:
    config = alembic_config(empty_database)
    command.upgrade(config, "head")
    assert {"agent_run", "trace_event"} <= table_names(empty_database)

    command.downgrade(config, "base")
    assert table_names(empty_database) == {"alembic_version"}

    command.upgrade(config, "head")
    assert {"agent_run", "trace_event"} <= table_names(empty_database)
