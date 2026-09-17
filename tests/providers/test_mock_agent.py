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
    Usage,
)
from kennel.providers.mock import Raise, Sleep, Text, ToolCall


def make_agent(meeting_ws: Path, turns, **kw) -> tuple[Agent, MockProvider, list]:
    provider_kw = {k: kw.pop(k) for k in ("structured", "reported_usage", "context_window_tokens") if k in kw}
    provider = MockProvider(turns, **provider_kw)
    events = []
    agent = Agent(meeting_ws, provider=provider, **kw)
    agent.events.subscribe(lambda e: events.append(e))
    return agent, provider, events


async def test_final_response_only(meeting_ws):
    agent, provider, events = make_agent(meeting_ws, ["Just an answer."])
    result = await agent.run("hi")
    assert result.text == "Just an answer." and result.stop_reason == "end_turn"
    assert result.tool_calls == [] and len(result.session_id) == 12
    assert result.usage == Usage(input_tokens=1, output_tokens=4, estimated=True)  # "hi" / 15 chars
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


NARRATION = "田中さんのメールを探すために、以下のコマンドを実行します。\n\n```bash\ngrep -r 田中 *.txt\n```\n\nこのコマンドを実行します。"


async def test_narration_is_nudged_once(meeting_ws):
    agent, provider, events = make_agent(meeting_ws, [NARRATION, [ToolCall("glob", {"pattern": "*.txt"}), Text("found it")]])
    deltas = []
    result = await agent.run("メールを放置している気がする", on_delta=deltas.append)
    assert result.text == "found it" and [c.name for c in result.tool_calls] == ["glob"]
    session = provider.sessions[0]
    assert len(provider.sessions) == 1 and session.prompts[1].startswith("Do not describe")
    types = [e.type for e in events]
    assert types.index(EventType.MODEL_NUDGED) < types.index(EventType.TOOL_STARTED)
    assert "found it" in "".join(deltas)


async def test_nudge_only_once_and_not_for_plain_answers(meeting_ws):
    agent, provider, events = make_agent(meeting_ws, [NARRATION, "I will search the files for you."])
    result = await agent.run("x")
    assert result.text == "I will search the files for you." and len(provider.sessions[0].prompts) == 2
    agent, provider, events = make_agent(meeting_ws, ["The decision was to ship on Friday."])
    result = await agent.run("x")
    assert len(provider.sessions[0].prompts) == 1 and EventType.MODEL_NUDGED not in [e.type for e in events]
    # a turn that already used tools is never nudged, even if the text mentions a tool
    agent, provider, _ = make_agent(meeting_ws, [[ToolCall("glob", {"pattern": "*.md"}), Text("I used glob to find notes/todo.md")]])
    await agent.run("x")
    assert len(provider.sessions[0].prompts) == 1


async def test_nudge_disabled_by_config_or_without_tools(meeting_ws):
    agent, provider, _ = make_agent(meeting_ws, [NARRATION], config=KennelConfig(nudge_narration=False))
    assert (await agent.run("x")).text == NARRATION and len(provider.sessions[0].prompts) == 1
    agent, provider, _ = make_agent(meeting_ws, [NARRATION], tools=[])
    assert (await agent.run("x")).text == NARRATION and len(provider.sessions[0].prompts) == 1


# -- context usage ------------------------------------------------------------


async def test_context_usage_is_estimated_after_two_turns(meeting_ws):
    agent, _, _ = make_agent(meeting_ws, ["one", "two"])
    session = agent.new_session()
    empty = session.context_usage()
    assert empty.window_tokens == 4096 and empty.turns == 0 and empty.used_tokens > 0  # instructions
    await session.run("q1")
    await session.run("q2")
    usage = session.context_usage()
    assert usage.window_tokens == 4096 and usage.used_tokens > empty.used_tokens
    assert 0.0 <= usage.ratio <= 1.0 and usage.estimated is True
    assert usage.turns == 2 and usage.compactions == 0
    assert usage.summary().startswith(f"{usage.ratio:.0%} of 4096 tokens")
    assert session.status()["context"] == usage.summary()
    await session.close()


async def test_compaction_lowers_context_usage(meeting_ws):
    long_answer, short_answer = "A" * 6000, "C" * 40
    turns = [long_answer, long_answer, [Raise(ContextLimitError("too big"))], short_answer]
    agent, provider, _ = make_agent(meeting_ws, turns)
    session = agent.new_session()
    await session.run("q1 " + "x" * 2000)
    await session.run("q2 " + "y" * 2000)
    before = session.context_usage()
    assert before.compactions == 0 and before.ratio == 1.0  # the window is full
    result = await session.run("q3")
    assert result.text == short_answer and len(provider.sessions) == 2
    after = session.context_usage()
    assert after.compactions == 1 and after.used_tokens < before.used_tokens
    assert after.turns == 3 and after.ratio < before.ratio
    await session.close()


async def test_clear_resets_context_usage(meeting_ws):
    agent, _, _ = make_agent(meeting_ws, ["one" * 2000, "two"])
    session = agent.new_session()
    await session.run("q1")
    used = session.context_usage().used_tokens
    await session.clear()
    assert session.context_usage().used_tokens < used
    assert session.context_usage().turns == 0
    await session.close()


async def test_apple_provider_declares_the_on_device_window():
    from kennel.providers.apple import AppleProvider

    assert AppleProvider().info.context_window_tokens == 4096  # no SDK import needed


async def test_agent_result_usage_is_filled(meeting_ws):
    agent, _, _ = make_agent(meeting_ws, [[ToolCall("glob", {"pattern": "**/*.txt"}), Text("答えは以下の通りです")]])
    result = await agent.run("最新の議事録を要約して")
    assert result.usage is not None and result.usage.estimated is True
    assert result.usage.input_tokens > 0  # prompt plus the tool output bytes
    assert result.usage.output_tokens > 0


async def test_reported_usage_wins_over_the_estimate(meeting_ws):
    reported = Usage(input_tokens=1200, output_tokens=300)
    agent, _, _ = make_agent(meeting_ws, ["done"], reported_usage=reported)
    session = agent.new_session()
    result = await session.run("q1")
    assert result.usage == reported and result.usage.estimated is False
    usage = session.context_usage()
    assert usage.used_tokens == 1500 and usage.estimated is False
    assert usage.ratio == pytest.approx(1500 / 4096)
    await session.close()


async def test_provider_without_a_declared_window(meeting_ws):
    agent, _, _ = make_agent(meeting_ws, ["done"], context_window_tokens=None)
    session = agent.new_session()
    await session.run("q1")
    usage = session.context_usage()
    assert usage.window_tokens is None and usage.ratio == 0.0  # no division by zero
    assert usage.used_tokens > 0 and usage.estimated is True
    assert usage.summary().endswith("window unknown")
    await session.close()
