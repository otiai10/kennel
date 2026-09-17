"""Agent/Session behaviour, driven deterministically by MockProvider."""

import asyncio
from pathlib import Path

import pytest

from kennel import (
    Agent,
    Approval,
    ContextLimitError,
    EventType,
    KennelConfig,
    MockProvider,
    PermissionKind,
    ProviderError,
    Tool,
    ToolParameter,
    ToolResult,
)
from kennel.providers.mock import Raise, Sleep, Text, ToolCall


def make_agent(meeting_ws: Path, turns, **kw) -> tuple[Agent, MockProvider, list]:
    provider = MockProvider(turns, structured=kw.pop("structured", None))
    events = []
    agent = Agent(meeting_ws, provider=provider, **kw)
    agent.events.subscribe(lambda e: events.append(e))
    return agent, provider, events


async def test_final_response_only(meeting_ws):
    agent, provider, events = make_agent(meeting_ws, ["Just an answer."])
    result = await agent.run("hi")
    assert result.text == "Just an answer." and result.stop_reason == "end_turn"
    assert result.tool_calls == [] and result.usage is None and len(result.session_id) == 12
    types = [e.type for e in events]
    assert types == [EventType.SESSION_STARTED, EventType.MODEL_STARTED, EventType.MODEL_COMPLETED, EventType.SESSION_COMPLETED]
    assert events[0].data["tools"] == ["edit", "glob", "grep", "read", "shell", "write"]
    assert provider.sessions[0].instructions.startswith("You are Kennel")
    assert str(meeting_ws.resolve()) in provider.sessions[0].instructions
    assert "Top-level entries: notes/ (1 files), transcripts/ (2 files)" in provider.sessions[0].instructions
    assert provider.sessions[0].closed


async def test_one_tool_call(meeting_ws):
    agent, provider, events = make_agent(meeting_ws, [[ToolCall("glob", {"pattern": "transcripts/*.txt"}), Text("two files")]])
    result = await agent.run("list")
    assert result.text == "two files"
    assert [c.status for c in result.tool_calls] == ["ok"]
    assert result.tool_calls[0].summary == "Glob transcripts/*.txt"
    assert provider.sessions[0].tool_results[0] == "transcripts/2026-09-15.txt\ntranscripts/2026-09-16.txt"
    types = [e.type for e in events]
    assert types[:6] == [EventType.SESSION_STARTED, EventType.MODEL_STARTED, EventType.TOOL_REQUESTED, EventType.TOOL_STARTED, EventType.TOOL_COMPLETED, EventType.MODEL_COMPLETED]
    completed = [e for e in events if e.type == EventType.TOOL_COMPLETED][0]
    assert completed.data["output_bytes"] > 0 and "content" not in completed.data


async def test_multiple_sequential_tool_calls(meeting_ws):
    turn = [
        ToolCall("glob", {"pattern": "**/*.txt"}),
        ToolCall("grep", {"pattern": "Decision", "path": "transcripts"}),
        ToolCall("read", {"path": "transcripts/2026-09-16.txt", "start_line": 1, "end_line": 4}),
        Text("done"),
    ]
    agent, provider, _ = make_agent(meeting_ws, [turn])
    result = await agent.run("go")
    assert [c.name for c in result.tool_calls] == ["glob", "grep", "read"]
    assert all(c.status == "ok" for c in result.tool_calls)
    assert "transcripts/2026-09-15.txt:6:" in provider.sessions[0].tool_results[1]
    assert provider.sessions[0].tool_results[2].startswith("     1| Kennel release planning")


async def test_permission_denied_without_prompter(meeting_ws):
    agent, provider, events = make_agent(meeting_ws, [[ToolCall("write", {"path": "minutes.md", "content": "x"}), Text("ok")]])
    result = await agent.run("write it")
    assert result.tool_calls[0].status == "denied"
    assert not (meeting_ws / "minutes.md").exists()
    assert provider.sessions[0].tool_results[0].startswith("Error: permission denied for write")
    assert EventType.PERMISSION_REQUESTED in [e.type for e in events] and EventType.PERMISSION_DENIED in [e.type for e in events]


async def test_permission_prompter_approves(meeting_ws):
    prompts = []

    def prompter(request):
        prompts.append(request)
        return Approval.ONCE

    turn = [ToolCall("write", {"path": "minutes.md", "content": "# M\n"}), ToolCall("edit", {"path": "minutes.md", "old_text": "# M", "new_text": "# Minutes"}), Text("ok")]
    agent, _, _ = make_agent(meeting_ws, [turn], prompter=prompter)
    result = await agent.run("write it")
    assert [c.status for c in result.tool_calls] == ["ok", "ok"]
    assert (meeting_ws / "minutes.md").read_text() == "# Minutes\n"
    assert prompts[0].tool_name == "write" and prompts[0].kind is PermissionKind.WRITE
    assert "Creates new file minutes.md" in prompts[0].details
    assert prompts[1].details.startswith("--- a/minutes.md")


async def test_allow_policy_and_read_only_tools(meeting_ws):
    turn = [ToolCall("write", {"path": "n.md", "content": "x"}), ToolCall("shell", {"command": "echo hi"}), Text("ok")]
    agent, _, _ = make_agent(meeting_ws, [turn], permissions={"write": "allow"})
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["ok", "denied"]
    agent, provider, _ = make_agent(meeting_ws, [list(turn)], tools=["glob", "grep", "read"])
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["invalid", "invalid"]
    assert provider.sessions[0].tool_results[0].startswith("Error: unknown tool 'write'")
    assert not (meeting_ws / "n.md").exists() or True


async def test_tool_failure_is_reported_to_model(meeting_ws):
    turn = [ToolCall("read", {"path": "missing.txt"}), ToolCall("read", {"path": "../outside.txt"}), ToolCall("read", {}), Text("recovered")]
    agent, provider, events = make_agent(meeting_ws, [turn])
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["error", "error", "invalid"]
    r = provider.sessions[0].tool_results
    assert r[0] == "Error: No such file: missing.txt"
    assert r[1].startswith("Error: Path escapes the workspace")
    assert "missing required argument 'path'" in r[2]
    assert result.text == "recovered"
    assert [e.type for e in events].count(EventType.TOOL_FAILED) == 3


async def test_repeated_identical_calls_blocked(meeting_ws):
    turn = [ToolCall("glob", {"pattern": "*.txt"})] * 5 + [ToolCall("glob", {"pattern": "*.md"}), Text("stop")]
    agent, provider, _ = make_agent(meeting_ws, [turn])
    result = await agent.run("loop")
    assert [c.status for c in result.tool_calls] == ["ok", "ok", "blocked", "blocked", "blocked", "ok"]
    assert "already made" in provider.sessions[0].tool_results[2]
    assert result.stop_reason == "end_turn"


async def test_tool_call_limit(meeting_ws):
    turn = [ToolCall("glob", {"pattern": f"*{i}*"}) for i in range(5)] + [Text("out of budget")]
    agent, provider, _ = make_agent(meeting_ws, [turn], config=KennelConfig(max_tool_calls=3))
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["ok", "ok", "ok", "blocked", "blocked"]
    assert result.stop_reason == "tool_limit"
    assert "limit" in provider.sessions[0].tool_results[3]


async def test_context_error_triggers_compaction_once(meeting_ws):
    turns = ["first answer", [Raise(ContextLimitError("too big"))], "answer after compaction"]
    agent, provider, events = make_agent(meeting_ws, turns)
    session = agent.new_session()
    await session.run("q1")
    result = await session.run("q2")
    assert result.text == "answer after compaction" and session.compactions == 1
    assert len(provider.sessions) == 2
    assert "Summary of the conversation so far" in provider.sessions[1].instructions
    assert "User asked: q1" in provider.sessions[1].instructions
    assert EventType.CONTEXT_COMPACTED in [e.type for e in events]
    assert provider.sessions[0].closed


async def test_context_error_without_history_or_twice_raises(meeting_ws):
    agent, _, events = make_agent(meeting_ws, [[Raise(ContextLimitError("x"))]])
    with pytest.raises(ContextLimitError, match="narrower"):
        await agent.run("huge")
    assert events[-1].type == EventType.SESSION_FAILED
    agent, _, _ = make_agent(meeting_ws, ["ok", [Raise(ContextLimitError("x"))], [Raise(ContextLimitError("x"))]])
    session = agent.new_session()
    await session.run("q1")
    with pytest.raises(ContextLimitError):
        await session.run("q2")


async def test_provider_exceptions_become_kennel_errors(meeting_ws):
    agent, _, _ = make_agent(meeting_ws, [[Raise(RuntimeError("boom"))]])
    with pytest.raises(ProviderError, match="boom"):
        await agent.run("x")
    agent, _, _ = make_agent(meeting_ws, [[Raise(ProviderError("custom"))]])
    with pytest.raises(ProviderError, match="custom"):
        await agent.run("x")


async def test_cancellation(meeting_ws):
    agent, provider, events = make_agent(meeting_ws, [[Sleep(5), Text("never")], "after cancel"])
    session = agent.new_session()
    task = asyncio.create_task(session.run("slow"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [e.type for e in events][-1] == EventType.MODEL_COMPLETED and events[-1].data["stop_reason"] == "cancelled"
    assert provider.sessions[0].closed
    result = await session.run("again")
    assert result.text == "after cancel" and len(provider.sessions) == 2 and session.history[-1].prompt == "again"


async def test_timeout(meeting_ws):
    agent, _, _ = make_agent(meeting_ws, [[Sleep(2), Text("late")]], config=KennelConfig(turn_timeout_seconds=0.2))
    result = await agent.run("slow")
    assert result.stop_reason == "timeout" and result.text == ""


async def test_streaming_deltas(meeting_ws):
    agent, _, events = make_agent(meeting_ws, [[ToolCall("glob", {"pattern": "*.md"}), Text("streamed answer text")]])
    deltas = []
    result = await agent.run("go", on_delta=deltas.append)
    assert "".join(deltas) == "streamed answer text" == result.text and len(deltas) > 1
    assert [e.type for e in events].count(EventType.MODEL_DELTA) == len(deltas)


async def test_multi_turn_session_and_clear(meeting_ws):
    agent, provider, _ = make_agent(meeting_ws, ["one", "two", "three"])
    session = agent.new_session()
    await session.run("a")
    await session.run("b")
    assert [t.response for t in session.history] == ["one", "two"] and len(provider.sessions) == 1
    assert session.status()["turns"] == 2 and session.status()["mode"] == "local"
    await session.clear()
    assert session.history == [] and provider.sessions[0].closed
    await session.run("c")
    assert len(provider.sessions) == 2 and "Summary of the conversation" not in provider.sessions[1].instructions
    await session.close()


async def test_custom_tool_and_extra_instructions(meeting_ws):
    class Shout(Tool):
        name = "shout"
        description = "upper-cases text"
        parameters = (ToolParameter("text", "string", "text"),)

        async def execute(self, arguments, context):
            return ToolResult(arguments["text"].upper(), metadata={"n": len(arguments["text"])})

        def summarize(self, arguments):
            return f"Shout {arguments['text']}"

    agent, provider, _ = make_agent(meeting_ws, [[ToolCall("shout", {"text": "hi"}), Text("HI")]], tools=["read", Shout()], instructions="Always shout.")
    result = await agent.run("x")
    assert provider.sessions[0].tool_results == ["HI"] and result.tool_calls[0].metadata == {"n": 2}
    assert provider.sessions[0].instructions.endswith("Always shout.")
    assert [t.name for t in provider.sessions[0].tools] == ["read", "shout"]


async def test_empty_prompt_and_unavailable(meeting_ws):
    from kennel import KennelError, ModelUnavailableError

    agent, _, _ = make_agent(meeting_ws, [])
    with pytest.raises(KennelError):
        await agent.run("   ")
    agent = Agent(meeting_ws, provider=MockProvider(available=False))
    with pytest.raises(ModelUnavailableError):
        agent.check_availability()


async def test_tool_output_is_bounded(meeting_ws):
    (meeting_ws / "huge.txt").write_text("x" * 100_000 + "\n")
    agent, provider, _ = make_agent(meeting_ws, [[ToolCall("read", {"path": "huge.txt"}), Text("ok")]], config=KennelConfig(max_tool_output_bytes=2000))
    result = await agent.run("go")
    assert result.tool_calls[0].truncated and result.tool_calls[0].output_bytes <= 2000
    assert provider.sessions[0].tool_results[0].endswith("[output truncated]")


async def test_structured_generation_passthrough(meeting_ws):
    agent, provider, _ = make_agent(meeting_ws, [], structured=[{"a": 1}])
    session = await agent.provider.create_session(instructions="i", tools=[], invoke=None)
    assert await session.respond_structured("p", {"type": "object"}) == {"a": 1}
    with pytest.raises(ProviderError):
        await session.respond_structured("p", {})


async def test_runner_cancel_turn_suppresses_events(meeting_ws):
    from kennel.runner import ToolRunner

    agent, _, events = make_agent(meeting_ws, [])
    runner = ToolRunner(agent.tools, agent.tool_context(), agent.events, "s")
    runner.begin_turn()
    runner.cancel_turn()
    out = await runner.invoke("glob", {"pattern": "*.md"})
    assert out.startswith("notes/todo.md") and runner.records[0].status == "ok"
    assert events == []  # cancelled turn: recorded, but nothing rendered
    runner.begin_turn()
    await runner.invoke("glob", {"pattern": "*.md"})
    assert [e.type for e in events] == [EventType.TOOL_REQUESTED, EventType.TOOL_STARTED, EventType.TOOL_COMPLETED]
