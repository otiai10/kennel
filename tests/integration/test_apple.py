"""Apple Foundation Models integration tests.

Run explicitly on an Apple Silicon Mac with Apple Intelligence enabled::

    KENNEL_APPLE_TESTS=1 pytest tests/integration -m apple

They assert structure and tool traces, never exact model wording.
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.apple,
    pytest.mark.skipif(os.environ.get("KENNEL_APPLE_TESTS") != "1", reason="set KENNEL_APPLE_TESTS=1 to run Apple integration tests"),
]

FIXTURE = Path(__file__).parent.parent / "fixtures" / "meeting_project"


@pytest.fixture(scope="module")
def provider():
    from kennel.providers.apple import AppleProvider

    p = AppleProvider()
    p.check_availability()
    return p


async def test_model_available(provider):
    assert provider.info.mode == "local"


async def test_simple_respond(provider):
    session = await provider.create_session(instructions="Answer with one word.", tools=[], invoke=None)
    text = await session.respond("Say hello.")
    assert isinstance(text, str) and text.strip()


async def test_streaming(provider):
    session = await provider.create_session(instructions="Be brief.", tools=[], invoke=None)
    deltas = [d async for d in session.stream("Count from 1 to 5, separated by commas.")]
    assert deltas and "5" in "".join(deltas)


async def test_structured_generation(provider):
    session = await provider.create_session(instructions="Extract facts.", tools=[], invoke=None)
    schema = {
        "type": "object",
        "properties": {"decisions": {"type": "array", "items": {"type": "string"}}, "title": {"type": "string"}},
        "required": ["decisions", "title"],
    }
    out = await session.respond_structured("Meeting: we decided to ship on Friday and to adopt pytest.", schema)
    assert isinstance(out.get("decisions"), list) and len(out["decisions"]) >= 1


async def test_structured_run_returns_a_dict(provider):
    """`Agent.run(schema=)` on the real model: a schema-shaped dict, tools still usable.

    Guided generation and tools coexist in one request (see spikes/06_structured_tools.py),
    so the agent keeps its tools here.
    """
    from kennel import Agent

    agent = Agent(FIXTURE, tools=["glob", "read"], provider=provider)
    schema = {
        "type": "object",
        "properties": {
            "decisions": {"type": "array", "items": {"type": "string"}, "description": "Decisions made"},
            "title": {"type": "string", "description": "Meeting title or topic"},
        },
        "required": ["decisions", "title"],
    }
    result = await agent.run(
        "Read transcripts/2026-09-16.txt and extract the meeting title and the decisions that were made.",
        schema=schema,
    )
    assert isinstance(result.structured_output, dict)
    assert set(schema["required"]) <= set(result.structured_output)
    assert isinstance(result.structured_output["decisions"], list)
    assert json.loads(result.text) == result.structured_output
    assert result.to_dict()["structured_output"] == result.structured_output


async def test_tool_calling_with_agent(provider):
    from kennel import Agent

    agent = Agent(FIXTURE, tools=["glob", "grep", "read"], provider=provider)
    result = await agent.run("Read transcripts/2026-09-16.txt and tell me the decision that was made.")
    names = [c.name for c in result.tool_calls]
    assert "read" in names and all(c.status in ("ok", "error", "blocked", "invalid") for c in result.tool_calls)
    assert any(c.name == "read" and "2026-09-16" in c.arguments.get("path", "") for c in result.tool_calls)
    assert result.text.strip()  # wording is the model's; only structure and trace are asserted


async def test_multi_step_reference_demo(provider):
    from kennel import Agent

    agent = Agent(FIXTURE, tools=["glob", "grep", "read"], provider=provider)
    result = await agent.run(
        "Find the latest meeting transcript in this workspace, read it, and list the decisions and TODOs."
    )
    assert len(result.tool_calls) >= 2
    assert result.tool_calls[0].name in ("glob", "grep")
    assert all("outside" not in c.arguments.get("path", "") for c in result.tool_calls)
    assert result.text.strip()


async def test_read_only_never_writes(provider, tmp_path):
    import shutil

    from kennel import Agent

    ws = tmp_path / "ws"
    shutil.copytree(FIXTURE, ws)
    before = sorted(p.relative_to(ws) for p in ws.rglob("*"))
    agent = Agent(ws, tools=["glob", "grep", "read"], provider=provider)
    await agent.run("Create a file called minutes.md summarizing transcripts/2026-09-16.txt.")
    assert sorted(p.relative_to(ws) for p in ws.rglob("*")) == before
