import asyncio

import pytest
from pydantic import ValidationError

from agentbench.domain.errors import ToolExecutionError
from agentbench.tools.mock.records import (
    DEFAULT_RECORDS,
    LookupRecordArgs,
    LookupRecordTool,
    RecordStore,
    SearchRecordsArgs,
    SearchRecordsTool,
)

STORE = RecordStore(DEFAULT_RECORDS)


def search(**kwargs: object) -> list[str]:
    output = asyncio.run(SearchRecordsTool(STORE).run(SearchRecordsArgs.model_validate(kwargs)))
    assert output["count"] == len(output["records"])
    return [r["id"] for r in output["records"]]


def test_search_is_case_insensitive_and_sorted_by_id() -> None:
    assert search(query="CABLE") == ["B-07", "B-09"]


def test_search_filters_by_category_and_limit() -> None:
    assert search(query="steel", category="hardware") == ["A-12", "A-15"]
    assert search(query="steel", limit=1) == ["A-12"]
    assert search(query="cable", category="hardware") == []


def test_search_with_no_match_returns_empty_list_not_error() -> None:
    assert search(query="spaceship") == []


@pytest.mark.parametrize(
    "bad",
    [{}, {"query": ""}, {"query": "a", "limit": 0}, {"query": "a", "limit": "5"}, {"q": "a"}],
)
def test_search_rejects_invalid_arguments(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SearchRecordsArgs.model_validate(bad)


def test_lookup_returns_the_record() -> None:
    output = asyncio.run(LookupRecordTool(STORE).run(LookupRecordArgs(record_id="A-12")))
    assert output["record"]["quantity"] == 120


def test_lookup_of_missing_record_is_a_permanent_tool_error() -> None:
    with pytest.raises(ToolExecutionError, match="not found") as info:
        asyncio.run(LookupRecordTool(STORE).run(LookupRecordArgs(record_id="Z-99")))
    assert info.value.retryable is False


@pytest.mark.parametrize("bad", ["a-12", "A12", "A-12; DROP TABLE", ""])
def test_lookup_rejects_malformed_ids(bad: str) -> None:
    with pytest.raises(ValidationError):
        LookupRecordArgs(record_id=bad)
