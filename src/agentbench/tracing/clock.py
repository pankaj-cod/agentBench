import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """Time source, injected so tests control time completely.

    `now` is wall-clock time, used only to say *when* something happened.
    `monotonic_us` never goes backwards and is used for every duration.
    """

    def now(self) -> datetime: ...
    def monotonic_us(self) -> int: ...
    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic_us(self) -> int:
        return time.monotonic_ns() // 1_000

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock:
    """Deterministic clock. Time moves only through `advance` and `sleep`, and `sleep`
    returns immediately, so tests of timeouts and backoff are instant and repeatable.
    Every `sleep` call is recorded in `sleeps`."""

    def __init__(self, start: datetime | None = None) -> None:
        self._start = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._elapsed_us = 0
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._start + timedelta(microseconds=self._elapsed_us)

    def monotonic_us(self) -> int:
        return self._elapsed_us

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time cannot go backwards")
        self._elapsed_us += round(seconds * 1_000_000)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)
