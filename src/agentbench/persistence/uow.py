from types import TracebackType
from typing import Self

from sqlalchemy.orm import Session, sessionmaker

from agentbench.persistence.repositories import AgentRunRepository


class UnitOfWork:
    """One database transaction. Nothing is saved unless `commit()` is called; leaving
    the `with` block rolls back anything uncommitted, including after an exception.

        with UnitOfWork(session_factory) as uow:
            uow.agent_runs.add(result)
            uow.commit()
    """

    agent_runs: AgentRunRepository

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def __enter__(self) -> Self:
        self._session = self._session_factory()
        self.agent_runs = AgentRunRepository(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._session.rollback()
        finally:
            self._session.close()

    def commit(self) -> None:
        self._session.commit()
