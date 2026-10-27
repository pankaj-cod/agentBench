import asyncio
from typing import Any, Literal

from agentbench.domain.errors import ToolExecutionError
from agentbench.tools.base import Tool

FaultMode = Literal["transient", "permanent", "hang"]


class FaultInjectingTool(Tool):
    """Wraps a tool and makes its first `failures` calls fail in a chosen way.

    - "transient": raises a retryable ToolExecutionError
    - "permanent": raises a non-retryable ToolExecutionError
    - "hang": sleeps for `hang_s`, so the registry's timeout fires

    After `failures` failed calls, calls go through to the wrapped tool. Used to test
    retries, timeouts and failure handling deterministically.
    """

    def __init__(
        self, inner: Tool, *, failures: int, mode: FaultMode, hang_s: float = 3600.0
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.description = inner.description
        self.args_model = inner.args_model
        self.failures = failures
        self.mode = mode
        self.hang_s = hang_s
        self.calls = 0

    async def run(self, args: Any) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            if self.mode == "hang":
                await asyncio.sleep(self.hang_s)
            raise ToolExecutionError(
                f"injected {self.mode} failure {self.calls}/{self.failures}",
                retryable=self.mode == "transient",
            )
        return await self.inner.run(args)
