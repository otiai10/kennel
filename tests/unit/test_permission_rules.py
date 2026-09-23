"""Permission rules ``tool(specifier)`` and the named permission modes (Issue #6)."""

import pytest

from kennel import (
    Agent,
    Approval,
    Decision,
    MockProvider,
    PermissionKind,
    PermissionManager,
    PermissionMode,
    Tool,
    ToolParameter,
    ToolResult,
)
from kennel.cli.main import BYPASS_WARNING, _header, build_agent, build_parser, resolve_mode
from kennel.errors import ConfigurationError
from kennel.permissions import BUILTIN_KINDS, READ_ONLY_TOOL_NAMES
from kennel.providers.mock import Text, ToolCall
from kennel.registry import READ_ONLY_TOOLS, builtin_registry
from kennel.rules import matches_text, normalize_path, parse_rule

SHELL = builtin_registry().create("shell")
WRITE = builtin_registry().create("write")
READ = builtin_registry().create("read")
GREP = builtin_registry().create("grep")


def decide(policy, tool, arguments, *, mode=PermissionMode.DEFAULT):
    pm = PermissionManager(policy, mode=mode)
    return pm.decision_for(tool.name, tool.permission, arguments, tool.match_rule)


# -- rule syntax -----------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        ("shell", ("shell", None)),
        ("shell(git *)", ("shell", "git *")),
        ("write(docs/**)", ("write", "docs/**")),
        ("  shell ( git * ) ", ("shell", "git *")),
        ("list_meetings", ("list_meetings", None)),
    ],
)
def test_parse_rule(key, expected):
    assert parse_rule(key) == expected


@pytest.mark.parametrize("key", ["shell(", "shell)", "(git *)", "shell()", "", "sh ell", "shell(a)b"])
def test_malformed_rule_is_a_configuration_error(key):
    """AC-6: a broken rule string is a ConfigurationError, not a silently ignored key."""
    with pytest.raises(ConfigurationError):
        parse_rule(key)
    with pytest.raises(ConfigurationError):
        PermissionManager({key: "allow"})


def test_invalid_decision_still_errors():
    with pytest.raises(ConfigurationError):
        PermissionManager({"shell(git *)": "maybe"})


# -- specifier matching ----------------------------------------------------


def test_shell_specifier_matches_the_command_line():
    """AC-1: shell=ask plus shell(git *)=allow lets git through and keeps the rest asking."""
    policy = {"shell": "ask", "shell(git *)": "allow"}
    assert decide(policy, SHELL, {"command": "git status"}) is Decision.ALLOW
    assert decide(policy, SHELL, {"command": "git log --oneline\n"}) is Decision.ALLOW
    assert decide(policy, SHELL, {"command": "ls"}) is Decision.ASK
    assert decide(policy, SHELL, {"command": "sudo git status"}) is Decision.ASK


def test_write_specifier_matches_the_path():
    """AC-2: write(docs/**) allows docs and leaves everything else asking."""
    policy = {"write": "ask", "write(docs/**)": "allow"}
    assert decide(policy, WRITE, {"path": "docs/a.md", "content": ""}) is Decision.ALLOW
    assert decide(policy, WRITE, {"path": "./docs/deep/a.md", "content": ""}) is Decision.ALLOW
    assert decide(policy, WRITE, {"path": "src/a.py", "content": ""}) is Decision.ASK


def test_a_path_rule_cannot_be_escaped_with_dotdot():
    """write(docs/**) must not cover a path that leaves docs again."""
    policy = {"write": "ask", "write(docs/**)": "allow"}
    assert decide(policy, WRITE, {"path": "docs/../src/evil.py", "content": ""}) is Decision.ASK
    assert decide(policy, WRITE, {"path": "docs/a/../b.md", "content": ""}) is Decision.ALLOW


def test_read_deny_rule_beats_the_read_default():
    assert decide({"read(**/.env)": "deny"}, READ, {"path": ".env"}) is Decision.DENY
    assert decide({"read(**/.env)": "deny"}, READ, {"path": "conf/.env"}) is Decision.DENY
    assert decide({"read(**/.env)": "deny"}, READ, {"path": "conf/app.yml"}) is Decision.ALLOW


def test_search_tools_match_their_search_root():
    policy = {"grep": "deny", "grep(src/**)": "allow"}
    assert decide(policy, GREP, {"pattern": "x", "path": "src/kennel"}) is Decision.DENY  # deny wins
    policy = {"grep(secret/**)": "deny"}
    assert decide(policy, GREP, {"pattern": "x", "path": "secret/keys"}) is Decision.DENY
    assert decide(policy, GREP, {"pattern": "x"}) is Decision.ALLOW  # default root "."


def test_custom_tool_falls_back_to_joined_arguments():
    class Reporter(Tool):
        name = "reporter"
        description = "report"
        permission = PermissionKind.WRITE
        parameters = (ToolParameter("action", "string", "what"),)

        async def execute(self, arguments, context):  # pragma: no cover - not executed here
            return ToolResult("ok")

    tool = Reporter()
    assert decide({"reporter(send*)": "deny"}, tool, {"action": "send-all"}) is Decision.DENY
    assert decide({"reporter(send*)": "deny"}, tool, {"action": "draft"}) is Decision.ASK


# -- precedence ------------------------------------------------------------


def test_deny_wins_over_allow_within_specifiers():
    """AC-3: a narrower deny rule still wins over a broader allow rule."""
    policy = {"shell(git *)": "allow", "shell(git push*)": "deny"}
    assert decide(policy, SHELL, {"command": "git push origin"}) is Decision.DENY
    assert decide(policy, SHELL, {"command": "git status"}) is Decision.ALLOW


def test_bare_deny_cannot_be_reopened_by_a_specifier():
    assert decide({"shell": "deny", "shell(git *)": "allow"}, SHELL, {"command": "git status"}) is Decision.DENY


def test_ask_beats_allow_among_matching_specifiers():
    policy = {"shell(git *)": "allow", "shell(* --force*)": "ask"}
    assert decide(policy, SHELL, {"command": "git push --force"}) is Decision.ASK


def test_last_rule_with_the_same_key_wins():
    pm = PermissionManager({"shell(git *)": "allow"})
    pm.update({"shell(git *)": "deny"})
    assert pm.decision_for("shell", PermissionKind.SHELL, {"command": "git status"}, SHELL.match_rule) is Decision.DENY
    assert [r.key for r in pm.rules()] == ["shell(git *)"]


def test_specifier_rules_do_not_leak_into_the_displayed_policy():
    pm = PermissionManager({"shell": "deny", "shell(git *)": "allow"})
    assert pm.policy()["shell"] is Decision.DENY
    assert [(r.key, r.decision) for r in pm.rules(specified_only=True)] == [("shell(git *)", Decision.ALLOW)]
    assert [r.key for r in pm.rules()] == ["shell", "shell(git *)"]


def test_builtin_kinds_match_the_registered_tools():
    """The name -> kind table used for display must not drift from the tools themselves."""
    registry = builtin_registry()
    assert {name: registry.create(name).permission for name in registry.names()} == BUILTIN_KINDS
    assert READ_ONLY_TOOL_NAMES == READ_ONLY_TOOLS


# -- modes -----------------------------------------------------------------

MODE_TABLE = {
    PermissionMode.READ_ONLY: (
        ("glob", "grep", "read"),
        {"glob": "allow", "grep": "allow", "read": "allow", "write": "deny", "edit": "deny", "shell": "deny", "web": "deny", "fetch": "deny"},
    ),
    PermissionMode.DEFAULT: (
        None,
        {"glob": "allow", "grep": "allow", "read": "allow", "write": "ask", "edit": "ask", "shell": "ask", "web": "deny", "fetch": "deny"},
    ),
    PermissionMode.ACCEPT_EDITS: (
        None,
        {"glob": "allow", "grep": "allow", "read": "allow", "write": "allow", "edit": "allow", "shell": "ask", "web": "ask", "fetch": "ask"},
    ),
    PermissionMode.DONT_ASK: (
        None,
        {"glob": "allow", "grep": "allow", "read": "allow", "write": "deny", "edit": "deny", "shell": "deny", "web": "deny", "fetch": "deny"},
    ),
    PermissionMode.BYPASS: (
        None,
        {"glob": "allow", "grep": "allow", "read": "allow", "write": "allow", "edit": "allow", "shell": "allow", "web": "allow", "fetch": "allow"},
    ),
}


@pytest.mark.parametrize("mode,expected", MODE_TABLE.items(), ids=lambda v: getattr(v, "value", ""))
def test_mode_tool_set_and_decisions(mode, expected, meeting_ws):
    """AC-4: each mode's tool set and per-tool decision match the specification table."""
    tools, policy = expected
    assert mode.tools() == tools
    assert {k: v.value for k, v in mode.policy().items()} == policy
    agent = Agent(meeting_ws, provider=MockProvider(), permission_mode=mode.value)
    assert sorted(agent.tools) == sorted(tools or ("glob", "grep", "read", "write", "edit", "shell"))
    for name, decision in policy.items():
        if name in agent.tools:
            assert agent.permissions.policy()[name].value == decision


def test_mode_applies_to_custom_tools_by_kind():
    pm = PermissionManager(mode=PermissionMode.DONT_ASK)
    assert pm.decision_for("custom_writer", PermissionKind.WRITE) is Decision.DENY
    assert pm.decision_for("custom_reader", PermissionKind.READ) is Decision.ALLOW
    assert PermissionManager(mode=PermissionMode.BYPASS).decision_for("custom_writer", PermissionKind.WRITE) is Decision.ALLOW


def test_invalid_mode():
    with pytest.raises(ConfigurationError):
        PermissionMode.parse("yolo")
    with pytest.raises(ConfigurationError):
        PermissionManager(mode="yolo")


def test_mode_is_the_base_and_rules_layer_on_top(meeting_ws):
    """AC-9: the mode supplies defaults, explicit rules override them."""
    agent = Agent(
        meeting_ws,
        provider=MockProvider(),
        permission_mode="accept-edits",
        permissions={"shell(git *)": "allow"},
    )
    pm = agent.permissions
    assert pm.decision_for("shell", PermissionKind.SHELL, {"command": "git status"}, SHELL.match_rule) is Decision.ALLOW
    assert pm.decision_for("shell", PermissionKind.SHELL, {"command": "ls"}, SHELL.match_rule) is Decision.ASK
    assert pm.policy()["write"] is Decision.ALLOW


def test_config_file_supplies_mode_and_rules(meeting_ws):
    (meeting_ws / "kennel.json").write_text(
        '{"permission_mode": "accept-edits", "permissions": {"shell(git *)": "allow"}}'
    )
    agent = Agent(meeting_ws, provider=MockProvider())
    assert agent.permission_mode is PermissionMode.ACCEPT_EDITS
    assert agent.permissions.policy()["write"] is Decision.ALLOW
    assert [r.key for r in agent.permissions.rules()] == ["shell(git *)"]


def test_config_file_rejects_a_bad_mode(meeting_ws):
    (meeting_ws / "kennel.json").write_text('{"permission_mode": "yolo"}')
    with pytest.raises(ConfigurationError):
        Agent(meeting_ws, provider=MockProvider())


# -- CLI flags are sugar for modes -----------------------------------------


def parse_args(argv):
    return build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "flag,mode",
    [("--read-only", "read-only"), ("--allow-write", "accept-edits"), ("--non-interactive", "dont-ask")],
)
def test_legacy_flags_produce_the_same_policy_as_their_mode(flag, mode, meeting_ws):
    """AC-5: the legacy flags are pure sugar, so the resulting policy is identical."""
    flagged = build_agent(parse_args([str(meeting_ws), "--provider", "mock", flag]), None)
    named = build_agent(parse_args([str(meeting_ws), "--provider", "mock", "--permission-mode", mode]), None)
    assert flagged.permission_mode is named.permission_mode is PermissionMode(mode)
    assert flagged.permissions.policy() == named.permissions.policy()
    assert sorted(flagged.tools) == sorted(named.tools)


def test_permission_mode_conflicts_with_the_legacy_flags(meeting_ws):
    with pytest.raises(ConfigurationError):
        resolve_mode(parse_args([str(meeting_ws), "--permission-mode", "bypass", "--read-only"]))
    with pytest.raises(ConfigurationError):
        resolve_mode(parse_args([str(meeting_ws), "--read-only", "--allow-write"]))
    assert resolve_mode(parse_args([str(meeting_ws)])) is PermissionMode.DEFAULT


def test_allow_shell_and_allow_web_stay_orthogonal(meeting_ws):
    agent = build_agent(parse_args([str(meeting_ws), "--provider", "mock", "--allow-shell", "--allow-web"]), None)
    assert agent.permissions.policy()["shell"] is Decision.ALLOW
    assert agent.permissions.policy()["web"] is Decision.ASK
    assert "web" in agent.tools
    bypass = build_agent(
        parse_args([str(meeting_ws), "--provider", "mock", "--permission-mode", "bypass", "--allow-web"]), None
    )
    assert bypass.permissions.policy()["web"] is Decision.ALLOW  # the mode is not downgraded


def test_bypass_header_warns(meeting_ws):
    """AC-7: the interactive header carries the bypass warning."""
    agent = build_agent(parse_args([str(meeting_ws), "--provider", "mock", "--permission-mode", "bypass"]), None)
    header = _header(agent)
    assert "permissions: bypass" in header
    assert BYPASS_WARNING in header
    plain = build_agent(parse_args([str(meeting_ws), "--provider", "mock"]), None)
    assert BYPASS_WARNING not in _header(plain)


# -- end to end through the runner ----------------------------------------


async def test_rules_decide_without_prompting_in_a_real_turn(meeting_ws):
    """AC-1 end to end: the allowed git command never reaches the prompter."""
    asked = []
    turn = [
        ToolCall("shell", {"command": "git status --short"}),
        ToolCall("shell", {"command": "echo hello"}),
        Text("done"),
    ]
    agent = Agent(
        meeting_ws,
        provider=MockProvider([turn]),
        permissions={"shell": "ask", "shell(git *)": "allow"},
        prompter=lambda request: (asked.append(request), Approval.DENY)[1],
    )
    result = await agent.run("go")
    assert result.text == "done"
    assert [c.status for c in result.tool_calls] == ["ok", "denied"]
    assert [r.tool_name for r in asked] == ["shell"]
    assert asked[0].arguments == {"command": "echo hello"}


async def test_deny_rule_blocks_the_call_in_a_real_turn(meeting_ws):
    """AC-3 end to end: git push is denied and nothing is prompted."""
    turn = [ToolCall("shell", {"command": "git push origin main"}), Text("stopped")]
    agent = Agent(
        meeting_ws,
        provider=MockProvider([turn]),
        permissions={"shell(git *)": "allow", "shell(git push*)": "deny"},
        prompter=lambda request: pytest.fail("must not prompt"),
    )
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["denied"]
    assert "permission denied" in result.tool_calls[0].error


async def test_path_rule_allows_docs_but_asks_elsewhere(meeting_ws):
    """AC-2 end to end: write(docs/**) writes without asking, src still asks."""
    turn = [
        ToolCall("write", {"path": "docs/notes.md", "content": "# notes\n"}),
        ToolCall("write", {"path": "src/x.py", "content": "x = 1\n"}),
        Text("done"),
    ]
    agent = Agent(
        meeting_ws,
        provider=MockProvider([turn]),
        permissions={"write": "ask", "write(docs/**)": "allow"},
    )
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["ok", "denied"]
    assert (meeting_ws / "docs" / "notes.md").read_text() == "# notes\n"
    assert not (meeting_ws / "src" / "x.py").exists()


# -- helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,text,expected",
    [
        ("git *", "git status", True),
        ("git *", "git", False),
        ("git*", "git", True),
        ("*", "anything at all", True),
        ("rm *", "cd x && rm -rf /", False),
        ("*/bin/*", "run /usr/bin/env", True),
    ],
)
def test_matches_text(pattern, text, expected):
    assert matches_text(pattern, text) is expected


@pytest.mark.parametrize(
    "value,expected",
    [("./docs/a.md", "docs/a.md"), ("docs//a.md", "docs/a.md"), ("", "."), (".", "."), ("/tmp/x", "/tmp/x")],
)
def test_normalize_path(value, expected):
    assert normalize_path(value) == expected
