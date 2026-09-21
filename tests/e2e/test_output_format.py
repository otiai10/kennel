"""`--output-format json|stream-json`: the machine-readable stdout contract.

Schema: docs/output-format.md.
"""

import json

from test_cli import TRANSCRIPT_FLOW, run_cli

from kennel import AgentResult, Usage
from kennel.runner import ToolCallRecord

RESULT_KEYS = {
    "type",
    "text",
    "stop_reason",
    "is_error",
    "error",
    "duration_ms",
    "session_id",
    "tool_calls",
    "usage",
    "structured_output",
    "compactions",
    "failure",
}

#: What `failure` carries when a turn failed (docs/output-format.md, `session.failed`).
FAILURE_KEYS = {
    "error",
    "error_type",
    "status",
    "provider_error",
    "tool_output_bytes",
    "tool_calls",
    "context_tokens_before",
    "context_window_tokens",
    "estimated",
    "duration_ms",
}

#: A turn that reads a transcript and then fails the way Apple's status 255 fails.
FAILING_FLOW = {
    "turns": [
        [
            {"tool": "read", "arguments": {"path": "transcripts/2026-09-16.txt"}},
            {"error": "Foundation Models error: Generation error (status: 255): None", "status": 255},
        ]
    ]
}


def test_json_prints_a_single_result_object(meeting_ws):
    p = run_cli(["-p", "summarize the latest meeting", "--output-format", "json"], meeting_ws)
    assert p.returncode == 0, p.stderr
    result = json.loads(p.stdout)  # a single JSON document, nothing else on stdout
    assert set(result) == RESULT_KEYS
    assert result["type"] == "result"
    assert result["stop_reason"] == "end_turn"
    assert result["is_error"] is False and result["error"] is None
    assert result["duration_ms"] > 0
    assert len(result["session_id"]) == 12
    assert result["text"].endswith("Unresolved: release notes owner.")
    assert result["compactions"] == 0 and result["usage"] is None and result["structured_output"] is None
    assert result["failure"] is None  # nothing failed
    names = [call["name"] for call in result["tool_calls"]]
    assert names == ["glob", "read", "read", "write"]
    assert [c["status"] for c in result["tool_calls"]] == ["ok", "ok", "error", "denied"]
    assert "● Glob" not in p.stdout  # human output is suppressed


def test_stream_json_emits_one_json_per_line(meeting_ws):
    p = run_cli(["-p", "x", "--output-format", "stream-json"], meeting_ws)
    assert p.returncode == 0, p.stderr
    lines = [json.loads(line) for line in p.stdout.splitlines()]
    types = [line["type"] for line in lines]
    assert types[0] == "session.started"
    assert types[-1] == "result"
    assert "tool.completed" in types and "permission.denied" in types
    assert types[-2] == "session.completed"  # closing events precede the result
    assert all({"type", "session_id", "timestamp", "data"} == set(line) for line in lines[:-1])
    session_ids = {line["session_id"] for line in lines}
    assert len(session_ids) == 1 and "" not in session_ids
    deltas = [line["data"]["text"] for line in lines if line["type"] == "model.delta"]
    assert deltas and "".join(deltas) == lines[-1]["text"]
    assert not any("content" in json.dumps(line["data"]) for line in lines if line["type"].startswith("tool."))


#: What `tool.completed` carries (docs/output-format.md, "Event objects").
TOOL_COMPLETED_KEYS = {
    "tool",
    "summary",
    "output_bytes",
    "estimated_tokens",
    "context_window_tokens",
    "window_exceeded",
    "withheld",
    "truncated",
    "duration_ms",
    "metadata",
}


def test_stream_json_tool_completed_reports_the_size_in_bytes_and_tokens(meeting_ws):
    """AC-4: the estimate and the window comparison are on the record, not only in the CLI."""
    (meeting_ws / "big.txt").write_text("x" * 20_000)
    script = {"turns": [[{"tool": "read", "arguments": {"path": "big.txt"}}, {"text": "done"}]]}
    p = run_cli(["-p", "read it", "--output-format", "stream-json"], meeting_ws, script=script)
    assert p.returncode == 0, p.stderr
    lines = [json.loads(line) for line in p.stdout.splitlines()]
    data = next(line["data"] for line in lines if line["type"] == "tool.completed")
    assert set(data) == TOOL_COMPLETED_KEYS
    assert data["output_bytes"] > 20_000
    assert data["estimated_tokens"] == -(-data["output_bytes"] // 4)  # ceil(bytes / 4)
    assert data["context_window_tokens"] == 4096 and data["window_exceeded"] is True
    assert data["withheld"] is True  # issue #47: it was measured, then kept out of the turn
    assert "(context" not in p.stdout  # the human lines stay out of a machine format


def test_stream_json_can_be_combined_with_trace(meeting_ws):
    p = run_cli(["-p", "x", "--output-format", "stream-json", "--trace"], meeting_ws)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout.splitlines()[-1])["type"] == "result"
    traced = [json.loads(line) for line in p.stderr.splitlines() if line.startswith("{")]
    assert traced and traced[0]["type"] == "session.started"


def test_json_reports_an_unavailable_model_as_an_error_result(meeting_ws):
    p = run_cli(["-p", "x", "--output-format", "json"], meeting_ws, script={"available": False})
    assert p.returncode == 3, p.stdout  # the exit code does not change with the format
    result = json.loads(p.stdout)
    assert result["type"] == "result" and result["is_error"] is True
    assert result["stop_reason"] == "error" and "unavailable" in result["error"]
    assert result["text"] == "" and result["tool_calls"] == []
    assert "Mock model is marked unavailable" in p.stderr


def test_json_reports_a_failed_turn_with_the_turn_facts(meeting_ws):
    """AC-4: the error record carries the same facts as `session.failed`."""
    p = run_cli(["-p", "x", "--output-format", "json"], meeting_ws, script=FAILING_FLOW)
    assert p.returncode == 1, p.stdout
    result = json.loads(p.stdout)
    assert result["is_error"] is True and result["stop_reason"] == "error"
    failure = result["failure"]
    assert set(failure) == FAILURE_KEYS
    assert failure["status"] == 255 and failure["error_type"] == "ProviderError"
    assert failure["tool_output_bytes"] > 0 and failure["tool_calls"] == 1
    assert failure["context_tokens_before"] > 0 and failure["context_window_tokens"] == 4096
    assert failure["estimated"] is True
    # stderr says the same in one human line
    assert "bytes of tool output" in p.stderr and "before the request" in p.stderr


def test_stream_json_ends_with_session_failed_then_the_error_result(meeting_ws):
    p = run_cli(["-p", "x", "--output-format", "stream-json"], meeting_ws, script=FAILING_FLOW)
    assert p.returncode == 1, p.stdout
    lines = [json.loads(line) for line in p.stdout.splitlines()]
    failed = [line for line in lines if line["type"] == "session.failed"]
    assert len(failed) == 1 and set(failed[0]["data"]) == FAILURE_KEYS
    assert lines[-1]["type"] == "result" and lines[-1]["failure"] == failed[0]["data"]


def test_json_reports_a_configuration_error_as_an_error_result(meeting_ws):
    p = run_cli(["-p", "x", "--output-format", "json", "--read-only", "--allow-write"], meeting_ws)
    assert p.returncode == 2
    result = json.loads(p.stdout)
    assert result["is_error"] is True and "cannot be combined" in result["error"]


def test_machine_format_requires_a_prompt(meeting_ws):
    for value in ("json", "stream-json"):
        p = run_cli(["--output-format", value], meeting_ws, script={"turns": []}, stdin="/exit\n")
        assert p.returncode == 2, p.stdout
        assert f"--output-format {value} needs -p/--prompt" in p.stderr
        assert p.stdout == ""  # the flag was rejected, so no JSON contract was established


def test_text_format_is_the_default_and_allowed_without_a_prompt(meeting_ws):
    explicit = run_cli(["--output-format", "text", "-p", "x"], meeting_ws)
    implicit = run_cli(["-p", "x"], meeting_ws)
    assert explicit.returncode == 0 and explicit.stdout == implicit.stdout
    interactive = run_cli(["--output-format", "text"], meeting_ws, script={"turns": []}, stdin="/exit\n")
    assert interactive.returncode == 0 and interactive.stdout.startswith("Kennel v")


def test_tool_limit_is_not_an_error(meeting_ws):
    (meeting_ws / "kennel.json").write_text(json.dumps({"agent": {"max_tool_calls": 2}}))
    p = run_cli(["-p", "x", "--output-format", "json"], meeting_ws, script=TRANSCRIPT_FLOW)
    assert p.returncode == 0, p.stderr
    result = json.loads(p.stdout)
    assert result["stop_reason"] == "tool_limit" and result["is_error"] is False


def test_to_dict_is_json_serializable_with_every_tool_call_field():
    record = ToolCallRecord("read", {"path": "a.txt"}, "Read a.txt", "ok", 12, True, 1.5, None, {"lines": 3})
    result = AgentResult(
        text="hi",
        stop_reason="end_turn",
        tool_calls=[record],
        usage=Usage(input_tokens=10, output_tokens=2),
        session_id="abc",
        duration_ms=12.3456,
        structured_output={"a": 1},
        compactions=2,
    )
    payload = result.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert set(payload) == RESULT_KEYS
    assert payload["duration_ms"] == 12.346  # rounded to milliseconds
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert payload["tool_calls"][0] == {
        "name": "read",
        "arguments": {"path": "a.txt"},
        "summary": "Read a.txt",
        "status": "ok",
        "output_bytes": 12,
        "truncated": True,
        "duration_ms": 1.5,
        "error": None,
        "metadata": {"lines": 3},
        "withheld": False,
    }
