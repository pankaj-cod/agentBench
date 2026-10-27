from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class Tool(ABC):
    """A capability an agent can invoke.

    `args_model` must be a Pydantic model with `extra="forbid"`; the ToolRegistry
    refuses to register a tool whose arguments are not strictly validated.
    `run` receives already-validated arguments and returns JSON-serialisable output,
    or raises a ToolError.
    """

    name: str
    description: str
    args_model: type[BaseModel]

    @abstractmethod
    async def run(self, args: Any) -> Any: ...

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }
