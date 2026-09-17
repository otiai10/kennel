"""Permission model: which tools may run, and how approval is obtained.

Policy values are ``allow``, ``ask`` or ``deny`` per tool name. ``ask`` needs a
prompter (an interactive UI); without one, ``ask`` means ``deny``. Approval can
be granted once or for the rest of the session. Nothing is persisted.

A prompter answers with an :class:`Approval` for the simple cases, or with
:class:`Allow` / :class:`Deny` when it wants to say more: a reason the model
should see, or corrected arguments. The same two types are what application
hooks return (:mod:`kennel.hooks`), so there is one vocabulary for "may this
call proceed" no matter who answers.

This layer is a usability/safety control, not an OS sandbox: an allowed
``shell`` tool can bypass the file tools' workspace boundary.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

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
class Allow:
    """Let the call proceed, optionally with corrected arguments.

    ``updated_arguments`` replaces the call's arguments (they are re-validated).
    ``remember="session"`` grants the tool for the rest of the session, like
    :attr:`Approval.SESSION`.
    """

    updated_arguments: Mapping[str, Any] | None = None
    remember: str | None = None

    @property
    def remember_session(self) -> bool:
        return self.remember == "session"


@dataclass(frozen=True)
class Deny:
    """Stop the call. ``message`` is what the model is told, so make it actionable."""

    message: str
    interrupt: bool = False


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    kind: PermissionKind
    summary: str
    details: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PermissionOutcome:
    """The resolved answer for one call, richer than a bool."""

    allowed: bool
    message: str | None = None
    updated_arguments: Mapping[str, Any] | None = None


Prompter = Callable[[PermissionRequest], "Approval | Allow | Deny"]

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

    def check(self, request: PermissionRequest, *, decision: Decision | None = None) -> bool:
        """Return True if the call may proceed, prompting if the policy says ``ask``."""
        return self.decide(request, decision=decision).allowed

    def decide(self, request: PermissionRequest, *, decision: Decision | None = None) -> PermissionOutcome:
        """Resolve one call, prompting if the policy says ``ask``.

        The prompter may answer with an :class:`Approval`, or with
        :class:`Allow` / :class:`Deny` to add a reason or corrected arguments;
        both reach the caller through :class:`PermissionOutcome`.
        """
        if decision is None:
            decision = self.decision_for(request.tool_name, request.kind)
        if decision is Decision.ALLOW:
            return PermissionOutcome(True)
        if decision is Decision.DENY:
            return PermissionOutcome(False)
        if self.has_session_grant(request.tool_name):
            return PermissionOutcome(True)
        if self._prompter is None:
            return PermissionOutcome(False)  # non-interactive: ask => deny
        return self.resolve(request.tool_name, self._prompter(request))

    def resolve(self, tool_name: str, answer: Approval | Allow | Deny | None) -> PermissionOutcome:
        """Interpret one answer about ``tool_name``, wherever it came from.

        Prompters and application hooks speak the same vocabulary, so what an
        ``Allow`` or a ``Deny`` *means* — including remembering a grant for the
        session — is decided here and nowhere else. ``None`` means "no opinion"
        and leaves the call allowed to continue.
        """
        if isinstance(answer, Deny):
            return PermissionOutcome(False, message=answer.message)
        if isinstance(answer, Allow):
            if answer.remember_session:
                self.grant_session(tool_name)
            return PermissionOutcome(True, updated_arguments=answer.updated_arguments)
        if answer is Approval.SESSION:
            self.grant_session(tool_name)
            return PermissionOutcome(True)
        if answer is None:
            return PermissionOutcome(True)
        return PermissionOutcome(answer is Approval.ONCE)
