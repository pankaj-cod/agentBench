from collections.abc import Iterable
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from agentbench.domain.errors import ToolExecutionError
from agentbench.tools.base import Tool

RecordId = Annotated[str, Field(pattern=r"^[A-Z]{1,3}-\d{1,4}$")]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: RecordId
    name: str
    category: str
    quantity: int = Field(ge=0)
    unit_price: float = Field(ge=0)


DEFAULT_RECORDS: tuple[Record, ...] = (
    Record(id="A-12", name="Steel bolt M8", category="hardware", quantity=120, unit_price=0.35),
    Record(id="A-15", name="Steel nut M8", category="hardware", quantity=300, unit_price=0.12),
    Record(id="B-07", name="USB-C cable 1m", category="cables", quantity=45, unit_price=6.5),
    Record(id="B-09", name="HDMI cable 2m", category="cables", quantity=18, unit_price=9.75),
    Record(id="C-03", name="Wireless mouse", category="peripherals", quantity=25, unit_price=19.99),
    Record(
        id="C-04", name="Mechanical keyboard", category="peripherals", quantity=8, unit_price=74.5
    ),
    Record(id="D-21", name="Laptop stand", category="accessories", quantity=0, unit_price=32.0),
)


class RecordStore:
    """Read-only, in-memory records. Results are always sorted by id, so every search
    returns the same order on every run."""

    def __init__(self, records: Iterable[Record]) -> None:
        self._by_id = {r.id: r for r in records}

    def get(self, record_id: str) -> Record | None:
        return self._by_id.get(record_id)

    def search(self, query: str, category: str | None, limit: int) -> list[Record]:
        needle = query.casefold()
        matches = [
            r
            for r in self._by_id.values()
            if needle in r.name.casefold() and (category is None or r.category == category)
        ]
        return sorted(matches, key=lambda r: r.id)[:limit]


class SearchRecordsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=100, description="Case-insensitive name text")
    category: str | None = Field(default=None, max_length=50)
    limit: int = Field(default=5, ge=1, le=20)


class SearchRecordsTool(Tool):
    name = "search_records"
    description = "Search inventory records whose name contains the query text."
    args_model = SearchRecordsArgs

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    async def run(self, args: SearchRecordsArgs) -> dict[str, Any]:
        records = self._store.search(args.query, args.category, args.limit)
        return {"count": len(records), "records": [r.model_dump() for r in records]}


class LookupRecordArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: RecordId


class LookupRecordTool(Tool):
    name = "lookup_record"
    description = "Fetch one inventory record by its id, e.g. A-12."
    args_model = LookupRecordArgs

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    async def run(self, args: LookupRecordArgs) -> dict[str, Any]:
        record = self._store.get(args.record_id)
        if record is None:
            raise ToolExecutionError(
                f"record not found: {args.record_id}",
                retryable=False,
                details={"record_id": args.record_id},
            )
        return {"record": record.model_dump()}
