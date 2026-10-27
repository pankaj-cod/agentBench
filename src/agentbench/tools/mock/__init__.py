from agentbench.tools.mock.calculator import CalculatorTool
from agentbench.tools.mock.records import (
    DEFAULT_RECORDS,
    LookupRecordTool,
    Record,
    RecordStore,
    SearchRecordsTool,
)
from agentbench.tools.registry import ToolRegistry


def default_registry(store: RecordStore | None = None) -> ToolRegistry:
    """A registry with the three standard mock tools over the default dataset."""
    store = store or RecordStore(DEFAULT_RECORDS)
    return ToolRegistry([SearchRecordsTool(store), LookupRecordTool(store), CalculatorTool()])


__all__ = [
    "DEFAULT_RECORDS",
    "CalculatorTool",
    "LookupRecordTool",
    "Record",
    "RecordStore",
    "SearchRecordsTool",
    "default_registry",
]
