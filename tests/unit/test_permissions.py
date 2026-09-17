import pytest

from kennel.errors import ConfigurationError
from kennel.permissions import (
    Approval,
    Decision,
    PermissionKind,
    PermissionManager,
    PermissionRequest,
    parse_policy,
)


def req(name="write", kind=PermissionKind.WRITE):
    return PermissionRequest(name, kind, f"{name} x")


def test_defaults():
    pm = PermissionManager()
    assert pm.decision_for("glob", PermissionKind.READ) is Decision.ALLOW
    assert pm.decision_for("write", PermissionKind.WRITE) is Decision.ASK
    assert pm.decision_for("shell", PermissionKind.SHELL) is Decision.ASK
    assert pm.decision_for("web", PermissionKind.WEB) is Decision.DENY
    assert pm.decision_for("custom_reader", PermissionKind.READ) is Decision.ALLOW
    assert pm.decision_for("custom_writer", PermissionKind.WRITE) is Decision.ASK


def test_non_interactive_ask_is_deny():
    pm = PermissionManager()
    assert pm.check(req("glob", PermissionKind.READ)) is True
    assert pm.check(req("write")) is False
    assert pm.check(req("shell", PermissionKind.SHELL)) is False
    assert not pm.interactive


def test_policy_overrides_and_invalid():
    pm = PermissionManager({"write": "allow", "glob": Decision.DENY})
    assert pm.check(req("write")) is True
    assert pm.check(req("glob", PermissionKind.READ)) is False
    with pytest.raises(ConfigurationError):
        parse_policy({"write": "maybe"})
    pm.set_decision("write", "deny")
    assert pm.check(req("write")) is False


def test_prompter_once_session_deny():
    answers = [Approval.ONCE, Approval.DENY, Approval.SESSION]
    calls = []

    def prompter(request):
        calls.append(request.tool_name)
        return answers.pop(0)

    pm = PermissionManager(prompter=prompter)
    assert pm.interactive
    assert pm.check(req()) is True  # once
    assert pm.check(req()) is False  # deny
    assert pm.check(req()) is True  # session
    assert pm.check(req()) is True  # no prompt: session grant
    assert calls == ["write", "write", "write"]
    assert pm.has_session_grant("write") and not pm.has_session_grant("edit")


def test_deny_never_prompts():
    pm = PermissionManager({"shell": "deny"}, prompter=lambda r: Approval.SESSION)
    assert pm.check(req("shell", PermissionKind.SHELL)) is False
