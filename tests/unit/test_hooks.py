"""before_tool / after_tool / before_prompt hooks and the richer prompter answers (Issue #7)."""

import pytest

from kennel import (
    Agent,
    Allow,
    Approval,
    Deny,
    EventType,
    HookError,
    HookMatcher,
    Hooks,
    KennelError,
    MockProvider,
    PermissionKind,
    PermissionManager,
    PermissionRequest,
    ToolResult,
)
from kennel.providers.mock import Text, ToolCall


def make_agent(meeting_ws, turns, **kw):
    provider = MockProvider(turns)
    events = []
    agent = Agent(meeting_ws, provider=provider, **kw)
    agent.events.subscribe(events.append)
    return agent, provider, events


GLOB = [ToolCall("glob", {"pattern": "*.md"}), Text("done")]


# -- before_tool -----------------------------------------------------------


async def test_before_tool_deny_blocks_the_call(meeting_ws):
    """AC-1: a denied call is not executed, is recorded blocked, and the model is told why."""

    def guard(call, ctx):
        assert call.name == "glob" and call.arguments == {"pattern": "*.md"}
        assert ctx.session_id and ctx.workspace.root == meeting_ws.resolve()
        assert ctx.permissions is agent.permissions and ctx.environment == {}
        assert ctx.tool.name == "glob" and ctx.tool_context.limits.max_output_bytes > 0
        return Deny("no")

    agent, provider, events = make_agent(meeting_ws, [GLOB], hooks=Hooks(before_tool=[guard]))
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["blocked"]
    assert result.tool_calls[0].error == "no"
    assert "no" in provider.sessions[0].tool_results[0]
    assert EventType.TOOL_BLOCKED in [e.type for e in events]
    assert EventType.TOOL_STARTED not in [e.type for e in events]


async def test_before_tool_allow_updates_arguments(meeting_ws):
    """AC-2: Allow(updated_arguments=...) is what the tool actually runs with."""

    def narrow(call, ctx):
        return Allow(updated_arguments={**call.arguments, "end_line": 2})

    turn = [ToolCall("read", {"path": "transcripts/2026-09-16.txt"}), Text("done")]
    agent, provider, _ = make_agent(meeting_ws, [turn], hooks=Hooks(before_tool=[narrow]))
    result = await agent.run("go")
    record = result.tool_calls[0]
    assert record.status == "ok"
    assert record.arguments == {"path": "transcripts/2026-09-16.txt", "end_line": 2}
    assert record.metadata["end_line"] == 2
    assert provider.sessions[0].tool_results[0].count("\n") <= 2


async def test_before_tool_updated_arguments_are_revalidated(meeting_ws):
    agent, provider, _ = make_agent(
        meeting_ws,
        [GLOB],
        hooks=Hooks(before_tool=[lambda call, ctx: Allow(updated_arguments={"pattern": None})]),
    )
    result = await agent.run("go")
    assert result.tool_calls[0].status == "invalid"
    assert provider.sessions[0].tool_results[0].startswith("Error: glob: missing required argument")


async def test_before_tool_allow_can_remember_the_session(meeting_ws):
    turn = [
        ToolCall("write", {"path": "a.md", "content": "x"}),
        ToolCall("write", {"path": "b.md", "content": "y"}),
        Text("done"),
    ]
    calls = []

    def approve(call, ctx):
        calls.append(call.name)
        return Allow(remember="session")

    agent, _, _ = make_agent(meeting_ws, [turn], hooks=Hooks(before_tool=[approve]))
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["ok", "ok"]  # no prompter, but the grant stands
    assert calls == ["write", "write"]
    # The grant went through PermissionManager, the same path a prompter answer takes.
    assert agent.permissions.has_session_grant("write")


async def test_before_tool_hooks_run_in_order_and_first_deny_wins(meeting_ws):
    seen = []

    def first(call, ctx):
        seen.append(("first", dict(call.arguments)))
        return Allow(updated_arguments={**call.arguments, "include_hidden": True})

    def second(call, ctx):
        seen.append(("second", dict(call.arguments)))
        return Deny("stop")

    def third(call, ctx):  # pragma: no cover - must not run
        pytest.fail("a hook after the Deny ran")

    agent, _, _ = make_agent(meeting_ws, [GLOB], hooks=Hooks(before_tool=[first, second, third]))
    result = await agent.run("go")
    assert result.tool_calls[0].status == "blocked"
    assert seen[0][0] == "first"
    assert seen[1] == ("second", {"pattern": "*.md", "include_hidden": True})


async def test_hook_matcher_limits_hooks_to_named_tools(meeting_ws):
    """AC-5: a HookMatcher(tools=["shell"]) hook does not see read calls."""
    seen = []
    hooks = Hooks(before_tool=[HookMatcher(tools=["shell"], hooks=[lambda call, ctx: seen.append(call.name)])])
    turn = [ToolCall("read", {"path": "transcripts/2026-09-16.txt"}), ToolCall("shell", {"command": "echo hi"}), Text("done")]
    agent, _, _ = make_agent(meeting_ws, [turn], hooks=hooks, permissions={"shell": "allow"})
    await agent.run("go")
    assert seen == ["shell"]


async def test_hooks_can_be_appended_after_construction(meeting_ws):
    agent, provider, _ = make_agent(meeting_ws, [GLOB])
    agent.hooks.before_tool.append(lambda call, ctx: Deny("late"))
    result = await agent.run("go")
    assert result.tool_calls[0].status == "blocked"
    assert "late" in provider.sessions[0].tool_results[0]


async def test_a_raising_hook_fails_the_call(meeting_ws):
    """AC-9: a broken policy is an error, not silence (unlike an event subscriber)."""

    def broken(call, ctx):
        raise RuntimeError("policy exploded")

    agent, provider, events = make_agent(meeting_ws, [GLOB], hooks=Hooks(before_tool=[broken]))
    result = await agent.run("go")
    assert result.tool_calls[0].status == "error"
    assert "policy exploded" in result.tool_calls[0].error
    assert provider.sessions[0].tool_results[0].startswith("Error: hook failed: policy exploded")
    assert EventType.TOOL_STARTED not in [e.type for e in events]


# -- after_tool ------------------------------------------------------------


async def test_after_tool_replaces_the_result(meeting_ws):
    """AC-3: the model sees the replacement, and the record reflects it."""

    def redact(call, result, ctx):
        assert call.name == "glob" and result.content
        return ToolResult("replaced", metadata={"redacted": True})

    agent, provider, _ = make_agent(meeting_ws, [GLOB], hooks=Hooks(after_tool=[redact]))
    result = await agent.run("go")
    assert provider.sessions[0].tool_results[0] == "replaced"
    assert result.tool_calls[0].metadata == {"redacted": True}
    assert result.tool_calls[0].status == "ok"


async def test_after_tool_returning_none_keeps_the_result(meeting_ws):
    agent, provider, _ = make_agent(meeting_ws, [GLOB], hooks=Hooks(after_tool=[lambda call, result, ctx: None]))
    await agent.run("go")
    assert provider.sessions[0].tool_results[0] != "replaced"


async def test_after_tool_replacement_is_still_bounded(meeting_ws):
    from kennel import KennelConfig

    config = KennelConfig(max_tool_output_bytes=64)
    hooks = Hooks(after_tool=[lambda call, result, ctx: ToolResult("x" * 500)])
    agent, provider, _ = make_agent(meeting_ws, [GLOB], hooks=hooks, config=config)
    result = await agent.run("go")
    assert result.tool_calls[0].truncated
    assert len(provider.sessions[0].tool_results[0]) < 500


# -- before_prompt ---------------------------------------------------------


async def test_before_prompt_adds_context(meeting_ws):
    """AC-4: what the hook returns reaches the provider's prompt."""
    agent, provider, _ = make_agent(
        meeting_ws, ["ok"], hooks=Hooks(before_prompt=[lambda prompt, session: "Today is 2026-09-18."])
    )
    result = await agent.run("what day is it")
    assert provider.sessions[0].prompts[0] == "what day is it\n\nToday is 2026-09-18."
    assert result.text == "ok"


async def test_before_prompt_sees_the_session_and_keeps_history_clean(meeting_ws):
    seen = []

    def hook(prompt, session):
        seen.append((prompt, session.id, len(session.history)))
        return f"turn {len(session.history)}"

    agent, provider, _ = make_agent(meeting_ws, ["one", "two"], hooks=Hooks(before_prompt=[hook]))
    session = agent.new_session()
    await session.run("first")
    await session.run("second")
    assert [p for p, _, _ in seen] == ["first", "second"]
    assert [n for _, _, n in seen] == [0, 1]
    assert {sid for _, sid, _ in seen} == {session.id}
    assert provider.sessions[0].prompts == ["first\n\nturn 0", "second\n\nturn 1"]
    assert [t.prompt for t in session.history] == ["first", "second"]  # history keeps the user's words


async def test_before_prompt_returning_none_changes_nothing(meeting_ws):
    agent, provider, _ = make_agent(meeting_ws, ["ok"], hooks=Hooks(before_prompt=[lambda prompt, session: None]))
    await agent.run("hi")
    assert provider.sessions[0].prompts == ["hi"]


async def test_a_raising_before_prompt_hook_fails_the_turn(meeting_ws):
    """#23 AC-1: the turn fails with HookError and says so through ``session.failed``.

    ``before_tool`` has a tool call to fail; ``before_prompt`` does not, so the turn is
    what fails. It must still arrive as a ``KennelError`` rather than the hook's own
    exception, because that is all a consumer (the CLI among them) catches.
    """

    def broken(prompt, session):
        raise RuntimeError("policy exploded")

    agent, provider, events = make_agent(meeting_ws, ["ok"], hooks=Hooks(before_prompt=[broken]))
    async with agent.new_session() as session:
        with pytest.raises(HookError) as caught:
            await session.run("hi")
        assert str(caught.value) == "hook failed: policy exploded"
        assert isinstance(caught.value, KennelError)
        assert isinstance(caught.value.__cause__, RuntimeError)  # the original is not lost
        failures = [e for e in events if e.type == EventType.SESSION_FAILED]
        assert len(failures) == 1
        assert failures[0].data["error"] == "hook failed: policy exploded"
        assert failures[0].data["error_type"] == "HookError"
        # The failed turn is on the record, and the prompt never reached the provider.
        assert session.last_failure["error"] == "hook failed: policy exploded"
        assert [turn.stop_reason for turn in session.history] == ["error"]
        assert provider.sessions == []


async def test_a_raising_before_prompt_hook_leaves_the_session_usable(meeting_ws):
    """#23 AC-3: no traceback and the next prompt is accepted.

    The CLI exposes no way to inject hooks (``kennel.cli.main`` never mentions them), so
    this criterion is met at the SDK level instead. What the REPL loop needs is exactly
    these two things: the error is a ``KennelError`` its ``except KennelError`` already
    catches, and the same session still serves the next prompt.
    """
    failing = [True]

    def flaky(prompt, session):
        if failing[0]:
            raise RuntimeError("policy exploded")
        return "context"

    agent, provider, _ = make_agent(meeting_ws, ["ok"], hooks=Hooks(before_prompt=[flaky]))
    async with agent.new_session() as session:
        with pytest.raises(HookError):
            await session.run("first")
        failing[0] = False
        result = await session.run("second")
    assert result.text == "ok"
    assert result.stop_reason == "end_turn"
    assert provider.sessions[-1].prompts == ["second\n\ncontext"]


async def test_a_raising_before_prompt_hook_fails_stream_after_the_event(meeting_ws):
    """#23 AC-2: through ``stream()``, ``session.failed`` is yielded before HookError."""

    def broken(prompt, session):
        raise RuntimeError("policy exploded")

    agent = Agent(meeting_ws, provider=MockProvider(["ok"]), hooks=Hooks(before_prompt=[broken]))
    seen = []
    async with agent.new_session() as session:
        with pytest.raises(HookError):
            async for event in session.stream("hi"):
                seen.append(event.type)
    # Not luck: _fail_turn puts session.failed on the queue before run() raises, and the
    # task's done callback only then puts the sentinel that ends the iteration.
    assert seen[-1] == EventType.SESSION_FAILED
    assert EventType.SESSION_COMPLETED not in seen  # the iterator ended on the failure


# -- prompter answers ------------------------------------------------------


async def test_prompter_deny_with_a_message(meeting_ws):
    """AC-6: Deny(message=...) denies the call and the reason reaches the model."""
    turn = [ToolCall("write", {"path": "a.md", "content": "x"}), Text("done")]
    agent, provider, events = make_agent(meeting_ws, [turn], prompter=lambda request: Deny(message="why"))
    result = await agent.run("go")
    assert result.tool_calls[0].status == "denied"
    assert "why" in provider.sessions[0].tool_results[0]
    assert result.tool_calls[0].error == "why"
    assert EventType.PERMISSION_DENIED in [e.type for e in events]
    assert not (meeting_ws / "a.md").exists()


async def test_prompter_allow_updates_arguments(meeting_ws):
    turn = [ToolCall("write", {"path": "a.md", "content": "x"}), Text("done")]
    agent, _, _ = make_agent(
        meeting_ws,
        [turn],
        prompter=lambda request: Allow(updated_arguments={"path": "a.md", "content": "corrected"}),
    )
    result = await agent.run("go")
    assert result.tool_calls[0].status == "ok"
    assert (meeting_ws / "a.md").read_text() == "corrected"


def test_prompter_allow_remembers_the_session():
    pm = PermissionManager(prompter=lambda request: Allow(remember="session"))
    request = PermissionRequest("write", PermissionKind.WRITE, "write a")
    assert pm.decide(request).allowed
    assert pm.has_session_grant("write")
    assert pm.check(request)


def test_plain_approvals_still_work():
    answers = [Approval.ONCE, Approval.DENY, Approval.SESSION]
    pm = PermissionManager(prompter=lambda request: answers.pop(0))
    request = PermissionRequest("write", PermissionKind.WRITE, "write a")
    assert [pm.check(request) for _ in range(3)] == [True, False, True]
    assert pm.has_session_grant("write")


def test_decide_reuses_a_decision_the_caller_resolved():
    from kennel import Decision

    pm = PermissionManager(prompter=lambda request: pytest.fail("must not prompt"))
    request = PermissionRequest("write", PermissionKind.WRITE, "write a")
    assert pm.decide(request, decision=Decision.ALLOW).allowed is True
    assert pm.decide(request, decision=Decision.DENY).allowed is False


# -- sync and async --------------------------------------------------------


async def test_async_and_sync_hooks_both_run(meeting_ws):
    """AC-7: callbacks may be coroutines or plain functions."""
    order = []

    async def async_before(call, ctx):
        order.append("async-before")
        return Allow(updated_arguments={**call.arguments, "include_hidden": True})

    def sync_before(call, ctx):
        order.append("sync-before")
        return None

    async def async_after(call, result, ctx):
        order.append("async-after")
        return ToolResult(result.content + "\n[checked]")

    async def async_prompt(prompt, session):
        order.append("async-prompt")
        return "extra"

    hooks = Hooks(
        before_tool=[async_before, sync_before],
        after_tool=[async_after],
        before_prompt=[async_prompt, lambda prompt, session: "more"],
    )
    agent, provider, _ = make_agent(meeting_ws, [GLOB], hooks=hooks)
    result = await agent.run("go")
    assert order == ["async-prompt", "async-before", "sync-before", "async-after"]
    assert result.tool_calls[0].arguments["include_hidden"] is True
    assert provider.sessions[0].tool_results[0].endswith("[checked]")
    assert provider.sessions[0].prompts[0] == "go\n\nextra\n\nmore"


async def test_no_hooks_is_the_old_behaviour(meeting_ws):
    agent, provider, _ = make_agent(meeting_ws, [GLOB])
    assert not agent.hooks
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["ok"]
    assert provider.sessions[0].prompts == ["go"]
