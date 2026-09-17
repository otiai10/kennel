"""Structured output through `Agent.run(schema=)` and the CLI's `--json-schema`."""

import json

import pytest
from test_cli import run_cli

from kennel import Agent, ConfigurationError, MockProvider
from kennel.cli.main import load_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "decisions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "decisions"],
}

VALUE = {"title": "Release planning", "decisions": ["Ship v0.1 on Friday"]}


async def test_structured_output_has_every_required_key(ws_dir):
    provider = MockProvider(structured=[VALUE])
    agent = Agent(ws_dir, tools=[], provider=provider)
    result = await agent.run("Extract the decisions", schema=SCHEMA)
    assert isinstance(result.structured_output, dict)
    assert all(key in result.structured_output for key in SCHEMA["required"])
    assert result.structured_output == VALUE
    assert json.loads(result.text) == VALUE  # text carries the same document
    assert result.stop_reason == "end_turn" and result.is_error is False
    assert result.to_dict()["structured_output"] == VALUE


async def test_tool_calls_are_recorded_alongside_the_schema(ws_dir):
    """Apple calls tools inside the guided-generation request; the mock mirrors that."""
    from kennel.providers.mock import ToolCall

    provider = MockProvider([[ToolCall("glob", {"pattern": "*.txt"})]], structured=[VALUE])
    agent = Agent(ws_dir, tools=["glob", "read"], provider=provider)
    result = await agent.run("Find the notes and extract the decisions", schema=SCHEMA)
    assert [call.name for call in result.tool_calls] == ["glob"]
    assert result.tool_calls[0].status == "ok"
    assert result.structured_output == VALUE


async def test_deltas_carry_the_document(ws_dir):
    deltas: list[str] = []
    agent = Agent(ws_dir, tools=[], provider=MockProvider(structured=[VALUE]))
    result = await agent.run("x", schema=SCHEMA, on_delta=deltas.append)
    assert "".join(deltas) == result.text


async def test_a_provider_without_structured_support_fails_loudly(ws_dir):
    from kennel import ProviderError

    agent = Agent(ws_dir, tools=[], provider=MockProvider(structured=[]))
    with pytest.raises(ProviderError):
        await agent.run("x", schema=SCHEMA)


def test_cli_prints_the_document(meeting_ws):
    (meeting_ws / "schema.json").write_text(json.dumps(SCHEMA))
    script = {"turns": [[{"tool": "glob", "arguments": {"pattern": "*.md"}}]], "structured": [VALUE]}
    p = run_cli(["-p", "x", "--json-schema", f"@{meeting_ws / 'schema.json'}"], meeting_ws, script=script)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout) == VALUE  # stdout is the document and nothing else
    assert "● Glob" not in p.stdout


def test_cli_inline_schema_and_json_output_format(meeting_ws):
    script = {"turns": [], "structured": [VALUE]}
    p = run_cli(
        ["-p", "x", "--json-schema", json.dumps(SCHEMA), "--output-format", "json"],
        meeting_ws,
        script=script,
    )
    assert p.returncode == 0, p.stderr
    result = json.loads(p.stdout)
    assert result["structured_output"] == VALUE
    assert json.loads(result["text"]) == VALUE


def test_cli_rejects_invalid_schema(meeting_ws):
    p = run_cli(["-p", "x", "--json-schema", "{not json"], meeting_ws)
    assert p.returncode == 2
    assert "--json-schema: invalid JSON" in p.stderr

    p = run_cli(["-p", "x", "--json-schema", "[1, 2]"], meeting_ws)
    assert p.returncode == 2
    assert "must be a JSON object" in p.stderr

    p = run_cli(["-p", "x", "--json-schema", "@/nope/missing.json"], meeting_ws)
    assert p.returncode == 2
    assert "cannot read" in p.stderr


def test_cli_json_schema_needs_a_prompt(meeting_ws):
    p = run_cli(["--json-schema", "{}"], meeting_ws, script={"turns": []}, stdin="/exit\n")
    assert p.returncode == 2
    assert "--json-schema needs -p/--prompt" in p.stderr
    assert p.stdout == ""


def test_load_schema_reads_inline_and_files(tmp_path):
    assert load_schema(None) is None
    assert load_schema('{"type": "object"}') == {"type": "object"}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(SCHEMA))
    assert load_schema(f"@{path}") == SCHEMA
    with pytest.raises(ConfigurationError):
        load_schema("nonsense")
