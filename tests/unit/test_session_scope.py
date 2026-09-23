"""What "allow for this session" remembers: the whole tool, or the tool's own scope (issue #78)."""

from typing import Any

from kennel import (
    Agent,
    Allow,
    Approval,
    Hooks,
    MockProvider,
    PermissionKind,
    PermissionManager,
    PermissionRequest,
    Tool,
    ToolParameter,
    ToolResult,
)
from kennel.providers.mock import Text, ToolCall


class Knock(Tool):
    """A tool whose grants are per door, like ``fetch`` is per host."""

    name = "knock"
    description = "knock on a door"
    permission = PermissionKind.WEB
    parameters = (ToolParameter("door", "string", "which door"),)

    def session_scope(self, arguments):
        return str(arguments.get("door"))

    async def execute(self, arguments: dict[str, Any], context) -> ToolResult:
        return ToolResult(f"knocked on {arguments['door']}")


ASK = {"knock": "ask"}


def knocks(*doors: str) -> list:
    return [[*(ToolCall("knock", {"door": d}) for d in doors), Text("done")]]


def test_a_scoped_grant_covers_only_its_scope():
    pm = PermissionManager({"knock": "ask"}, prompter=lambda request: Approval.SESSION)
    asked = PermissionRequest("knock", PermissionKind.WEB, "knock a", scope="a")
    assert pm.decide(asked).allowed
    assert pm.has_session_grant("knock", "a")
    assert not pm.has_session_grant("knock", "b") and not pm.has_session_grant("knock")
    assert pm.session_scopes("knock") == ["a"]


def test_an_unscoped_grant_still_covers_the_whole_tool():
    pm = PermissionManager()
    pm.grant_session("write")
    assert pm.has_session_grant("write") and pm.has_session_grant("write", "anything")
    assert pm.session_scopes("write") == [None]


async def test_the_prompter_is_asked_again_for_another_scope(meeting_ws):
    asked: list[PermissionRequest] = []

    def prompter(request):
        asked.append(request)
        return Approval.SESSION

    agent = Agent(meeting_ws, provider=MockProvider(knocks("a", "a", "b")), tools=[Knock()], prompter=prompter, permissions=ASK)
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["ok", "ok", "ok"]
    assert [r.scope for r in asked] == ["a", "b"]  # the second "a" went through on the grant


async def test_a_hook_session_grant_uses_the_same_scope(meeting_ws):
    asked: list[str | None] = []

    def prompter(request):
        asked.append(request.scope)
        return Approval.ONCE

    def approve_a(call, ctx):
        return Allow(remember="session") if call.arguments["door"] == "a" else None

    agent = Agent(
        meeting_ws,
        provider=MockProvider(knocks("a", "b")),
        tools=[Knock()],
        permissions=ASK,
        prompter=prompter,
        hooks=Hooks(before_tool=[approve_a]),
    )
    await agent.run("go")
    assert agent.permissions.session_scopes("knock") == ["a"]
    assert asked == ["b"]  # "a" was covered by the hook's grant, "b" was not


async def test_other_tools_keep_their_tool_wide_grant(meeting_ws):
    """write (and web, and every tool that does not narrow it) is still granted as a whole."""
    asked: list[PermissionRequest] = []

    def prompter(request):
        asked.append(request)
        return Approval.SESSION

    turn = [
        ToolCall("write", {"path": "a.md", "content": "x"}),
        ToolCall("write", {"path": "b.md", "content": "y"}),
        Text("done"),
    ]
    agent = Agent(meeting_ws, provider=MockProvider([turn]), prompter=prompter, tools=["write"])
    await agent.run("go")
    assert [r.scope for r in asked] == [None]
    assert agent.permissions.has_session_grant("write")
