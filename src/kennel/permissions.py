"""Permission model: which tools may run, and how approval is obtained.

Policy values are ``allow``, ``ask`` or ``deny`` per tool name. ``ask`` needs a
prompter (an interactive UI); without one, ``ask`` means ``deny``. Approval can
be granted once or for the rest of the session. Nothing is persisted.

This layer is a usability/safety control, not an OS sandbox: an allowed
``shell`` tool can bypass the file tools' workspace boundary.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

from .errors import ConfigurationError


class PermissionKind(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    WEB = "web"


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class Approval(str, Enum):
    ONCE = "once"
    SESSION = "session"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    kind: PermissionKind
    summary: str
    details: str | None = None
    warnings: tuple[str, ...] = ()


Prompter = Callable[[PermissionRequest], Approval]

DEFAULT_POLICY: dict[str, Decision] = {
    "glob": Decision.ALLOW,
    "grep": Decision.ALLOW,
    "read": Decision.ALLOW,
    "write": Decision.ASK,
    "edit": Decision.ASK,
    "shell": Decision.ASK,
    "web": Decision.DENY,
}

_KIND_DEFAULT: dict[PermissionKind, Decision] = {
    PermissionKind.READ: Decision.ALLOW,
    PermissionKind.WRITE: Decision.ASK,
    PermissionKind.SHELL: Decision.ASK,
    PermissionKind.WEB: Decision.DENY,
}


def parse_policy(values: Mapping[str, Decision | str] | None) -> dict[str, Decision]:
    policy: dict[str, Decision] = {}
    for name, value in (values or {}).items():
        try:
            policy[name] = value if isinstance(value, Decision) else Decision(str(value).lower())
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid permission {value!r} for tool {name!r}; expected allow, ask or deny"
            ) from exc
    return policy


class PermissionManager:
    """Decides whether a tool call may proceed. Safe to call from any thread."""

    def __init__(
        self,
        policy: Mapping[str, Decision | str] | None = None,
        *,
        prompter: Prompter | None = None,
    ) -> None:
        self._policy: dict[str, Decision] = dict(DEFAULT_POLICY)
        self._policy.update(parse_policy(policy))
        self._prompter = prompter
        self._session_grants: set[str] = set()
        self._lock = threading.Lock()

    @property
    def interactive(self) -> bool:
        return self._prompter is not None

    def cancel_prompt(self) -> None:
        """Release a prompter that is waiting for input, if it supports being cancelled.

        Called when a turn is interrupted: a prompter blocked on stdin would
        otherwise keep the turn alive until the user answers it.
        """
        cancel = getattr(self._prompter, "cancel", None)
        if callable(cancel):
            cancel()

    def policy(self) -> dict[str, Decision]:
        return dict(self._policy)

    def decision_for(self, tool_name: str, kind: PermissionKind) -> Decision:
        with self._lock:
            if tool_name in self._policy:
                return self._policy[tool_name]
        return _KIND_DEFAULT[kind]

    def set_decision(self, tool_name: str, decision: Decision | str) -> None:
        with self._lock:
            self._policy[tool_name] = parse_policy({tool_name: decision})[tool_name]

    def grant_session(self, tool_name: str) -> None:
        with self._lock:
            self._session_grants.add(tool_name)

    def has_session_grant(self, tool_name: str) -> bool:
        with self._lock:
            return tool_name in self._session_grants

    def check(self, request: PermissionRequest) -> bool:
        """Return True if the call may proceed, prompting if the policy says ``ask``."""
        decision = self.decision_for(request.tool_name, request.kind)
        if decision is Decision.ALLOW:
            return True
        if decision is Decision.DENY:
            return False
        if self.has_session_grant(request.tool_name):
            return True
        if self._prompter is None:
            return False  # non-interactive: ask => deny
        approval = self._prompter(request)
        if approval is Approval.SESSION:
            self.grant_session(request.tool_name)
            return True
        return approval is Approval.ONCE
