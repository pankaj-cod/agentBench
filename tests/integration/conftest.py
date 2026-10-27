"""PostgreSQL fixtures.

Strategy:
- AGENTBENCH_TEST_DATABASE_URL points at a server the tests may create databases on.
  If it is not set, these tests are skipped with a reason, unless
  AGENTBENCH_REQUIRE_POSTGRES=1 (set in CI), in which case they fail instead of
  being silently skipped.
- Once per test session, a uniquely named database is created, migrated with Alembic
  to the latest revision (so migrations are exercised, not `create_all`), and dropped
  at the end.
- Each test runs inside an outer transaction that is always rolled back. Sessions join
  it with savepoints, so code under test can call commit() and tests stay isolated.
"""

import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import URL, Engine, create_engine, make_url, text
from sqlalchemy.orm import Session, sessionmaker

from agentbench.persistence.uow import UnitOfWork

TEST_URL_ENV = "AGENTBENCH_TEST_DATABASE_URL"
REQUIRE_ENV = "AGENTBENCH_REQUIRE_POSTGRES"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def server_url() -> URL:
    raw = os.environ.get(TEST_URL_ENV)
    if not raw:
        message = f"{TEST_URL_ENV} is not set; PostgreSQL tests need a server"
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(message)
        pytest.skip(message)
    return make_url(raw)


@contextmanager
def temporary_database(server: URL) -> Iterator[URL]:
    """Create an empty database with a random name; drop it afterwards."""
    name = f"agentbench_test_{uuid.uuid4().hex[:12]}"  # generated, safe to interpolate
    admin = create_engine(server, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        yield server.set(database=name)
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def alembic_config(url: URL) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    rendered = url.render_as_string(hide_password=False).replace("%", "%%")
    config.set_main_option("sqlalchemy.url", rendered)
    return config


@pytest.fixture
def empty_database(server_url: URL) -> Iterator[URL]:
    with temporary_database(server_url) as url:
        yield url


@pytest.fixture(scope="session")
def migrated_engine(server_url: URL) -> Iterator[Engine]:
    with temporary_database(server_url) as url:
        command.upgrade(alembic_config(url), "head")
        engine = create_engine(url)
        yield engine
        engine.dispose()


@pytest.fixture
def session_factory(migrated_engine: Engine) -> Iterator[sessionmaker[Session]]:
    connection = migrated_engine.connect()
    outer = connection.begin()
    yield sessionmaker(
        bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    outer.rollback()
    connection.close()


@pytest.fixture
def uow_factory(session_factory: sessionmaker[Session]) -> Callable[[], UnitOfWork]:
    return lambda: UnitOfWork(session_factory)
