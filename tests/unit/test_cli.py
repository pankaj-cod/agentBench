import io
import json

from agentbench.cli import main


def run_cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def test_successful_run_prints_trace_and_exits_zero() -> None:
    code, output = run_cli("run", "What is the quantity of record A-12?")
    assert code == 0
    assert "status   COMPLETED (COMPLETED)" in output
    assert "answer   120" in output
    assert "TOOL_CALL_SUCCEEDED" in output


def test_failed_run_exits_one_and_shows_the_reason() -> None:
    code, output = run_cli("run", "Write a poem")
    assert code == 1
    assert "AGENT_ERROR" in output
    assert "UnsupportedTaskError" in output


def test_json_output_is_the_full_result() -> None:
    code, output = run_cli("run", "Compute 6 * 7", "--tools", "calculator", "--json")
    data = json.loads(output)
    assert code == 0
    assert data["final_answer"] == "42"
    assert [e["seq"] for e in data["events"]] == list(range(len(data["events"])))
