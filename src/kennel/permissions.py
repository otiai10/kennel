"""Permission model: which tools may run, and how approval is obtained.

A policy maps **rules** to ``allow`` / ``ask`` / ``deny``. A rule is a tool name
(``shell``) or a tool name with a specifier (``shell(git *)``, ``write(docs/**)``);
what a specifier means is decided by the tool itself
(:meth:`kennel.Tool.match_rule`). ``ask`` needs a prompter (an interactive UI);
without one, ``ask`` means ``deny``. Approval can be granted once or for the rest
of the session. Nothing is persisted.

A :class:`PermissionMode` supplies the defaults that rules are layered on top of,
so ``read-only`` / ``default`` / ``accept-edits`` / ``dont-ask`` / ``bypass`` can
be named instead of spelled out per tool.

This layer is a usability/safety control, not an OS sandbox: an allowed
``shell`` tool can bypass the file tools' workspace boundary.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import ConfigurationError
from .rules import match_arguments, parse_rule


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


BUILTIN_KINDS: dict[str, PermissionKind] = {
    "glob": PermissionKind.READ,
    "grep": PermissionKind.READ,
    "read": PermissionKind.READ,
    "write": PermissionKind.WRITE,
    "edit": PermissionKind.WRITE,
    "shell": PermissionKind.SHELL,
    "web": PermissionKind.WEB,
}

_A, _Q, _D = Decision.ALLOW, Decision.ASK, Decision.DENY

_MODE_KIND_DEFAULTS: dict[str, dict[PermissionKind, Decision]] = {
    # read: write: shell: web:
    "read-only": {PermissionKind.READ: _A, PermissionKind.WRITE: _D, PermissionKind.SHELL: _D, PermissionKind.WEB: _D},
    "default": {PermissionKind.READ: _A, PermissionKind.WRITE: _Q, PermissionKind.SHELL: _Q, PermissionKind.WEB: _D},
    "accept-edits": {PermissionKind.READ: _A, PermissionKind.WRITE: _A, PermissionKind.SHELL: _Q, PermissionKind.WEB: _Q},
    "dont-ask": {PermissionKind.READ: _A, PermissionKind.WRITE: _D, PermissionKind.SHELL: _D, PermissionKind.WEB: _D},
    "bypass": {PermissionKind.READ: _A, PermissionKind.WRITE: _A, PermissionKind.SHELL: _A, PermissionKind.WEB: _A},
}

_READ_ONLY_TOOL_NAMES: tuple[str, ...] = ("glob", "grep", "read")


class PermissionMode(str, Enum):
    """A named set of permission defaults.

    ``read-only`` also restricts which tools exist at all; the others only
    change decisions. ``dont-ask`` is ``default`` with every ``ask`` turned into
    ``deny`` (nothing can prompt), ``bypass`` allows everything and is meant for
    non-interactive runs where the caller has accepted the risk.
    """

    READ_ONLY = "read-only"
    DEFAULT = "default"
    ACCEPT_EDITS = "accept-edits"
    DONT_ASK = "dont-ask"
    BYPASS = "bypass"

    @classmethod
    def parse(cls, value: PermissionMode | str) -> PermissionMode:
        if isinstance(value, PermissionMode):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid permission mode {value!r}; expected one of {', '.join(m.value for m in cls)}"
            ) from exc

    def tools(self) -> tuple[str, ...] | None:
        """Tool names this mode restricts the agent to, or ``None`` for no restriction."""
        return _READ_ONLY_TOOL_NAMES if self is PermissionMode.READ_ONLY else None

    def kind_defaults(self) -> dict[PermissionKind, Decision]:
        """Default decision per permission kind, used for tools the policy does not name."""
        return dict(_MODE_KIND_DEFAULTS[self.value])

    def policy(self) -> dict[str, Decision]:
        """The mode expressed as a decision per built-in tool name."""
        defaults = self.kind_defaults()
        return {name: defaults[kind] for name, kind in BUILTIN_KINDS.items()}


DEFAULT_MODE = PermissionMode.DEFAULT
DEFAULT_POLICY: dict[str, Decision] = DEFAULT_MODE.policy()


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    kind: PermissionKind
    summary: str
    details: str | None = None
    warnings: tuple[str, ...] = ()
    # Validated tool arguments, so rule specifiers and application hooks can see
    # what is actually being asked for. Excluded from equality so the request
    # stays hashable.
    arguments: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class PermissionRule:
    """One parsed policy entry: ``shell(git *) -> allow``."""

    tool_name: str
    specifier: str | None
    decision: Decision

    @property
    def key(self) -> str:
        return self.tool_name if self.specifier is None else f"{self.tool_name}({self.specifier})"


Prompter = Callable[[PermissionRequest], Approval]
#: Decides whether a rule specifier covers a call; normally ``Tool.match_rule``.
RuleMatcher = Callable[[str, Mapping[str, Any]], bool]


def parse_policy(values: Mapping[str, Decision | str] | None) -> dict[str, Decision]:
    """Validate a policy mapping, keeping the original rule keys.

    Both sides are checked: the key must be ``tool`` or ``tool(specifier)`` and
    the value must be a :class:`Decision`.
    """
    policy: dict[str, Decision] = {}
    for name, value in (values or {}).items():
        parse_rule(name)  # raises ConfigurationError on a malformed rule
        try:
            policy[name] = value if isinstance(value, Decision) else Decision(str(value).lower())
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid permission {value!r} for rule {name!r}; expected allow, ask or deny"
            ) from exc
    return policy


def parse_rules(values: Mapping[str, Decision | str] | None) -> list[PermissionRule]:
    """Parse a policy mapping into rules, preserving order."""
    rules: list[PermissionRule] = []
    for key, decision in parse_policy(values).items():
        tool_name, specifier = parse_rule(key)
        rules.append(PermissionRule(tool_name, specifier, decision))
    return rules


class PermissionManager:
    """Decides whether a tool call may proceed. Safe to call from any thread."""

    def __init__(
        self,
        policy: Mapping[str, Decision | str] | None = None,
        *,
        prompter: Prompter | None = None,
        mode: PermissionMode | str = DEFAULT_MODE,
    ) -> None:
        self.mode = PermissionMode.parse(mode)
        self._defaults = self.mode.policy()
        self._kind_defaults = self.mode.kind_defaults()
        self._policy: dict[str, Decision] = {}
        self._rules: list[PermissionRule] = []
        self._prompter = prompter
        self._session_grants: set[str] = set()
        self._lock = threading.Lock()
        self.update(policy)

    @property
    def interactive(self) -> bool:
        return self._prompter is not None

    def policy(self) -> dict[str, Decision]:
        """Effective decision per tool name, ignoring specifier rules (for display)."""
        with self._lock:
            return {**self._defaults, **self._policy}

    def rules(self) -> list[PermissionRule]:
        """Specifier rules in declaration order."""
        with self._lock:
            return list(self._rules)

    def update(self, policy: Mapping[str, Decision | str] | None) -> None:
        """Layer more rules on top of the current policy (later wins)."""
        for rule in parse_rules(policy):
            with self._lock:
                if rule.specifier is None:
                    self._policy[rule.tool_name] = rule.decision
                else:
                    self._rules = [r for r in self._rules if r.key != rule.key] + [rule]

    def decision_for(
        self,
        tool_name: str,
        kind: PermissionKind,
        arguments: Mapping[str, Any] | None = None,
        matcher: RuleMatcher | None = None,
    ) -> Decision:
        """Resolve the decision for one call.

        ``deny`` always wins, whether it comes from ``shell`` or ``shell(rm *)``.
        Otherwise a matching specifier rule beats the bare tool rule, and within
        one level ``ask`` beats ``allow``. With no rule at all the mode's default
        for ``kind`` applies.
        """
        with self._lock:
            bare = self._policy.get(tool_name)
            candidates = [r for r in self._rules if r.tool_name == tool_name]
            kind_default = self._kind_defaults[kind]
        matched: set[Decision] = set()
        if candidates and arguments is not None:
            match = matcher or match_arguments
            for rule in candidates:
                if rule.specifier is not None and match(rule.specifier, arguments):
                    matched.add(rule.decision)
        if Decision.DENY in matched or bare is Decision.DENY:
            return Decision.DENY
        if matched:
            return Decision.ASK if Decision.ASK in matched else Decision.ALLOW
        return bare if bare is not None else kind_default

    def set_decision(self, tool_name: str, decision: Decision | str) -> None:
        self.update({tool_name: decision})

    def grant_session(self, tool_name: str) -> None:
        with self._lock:
            self._session_grants.add(tool_name)

    def has_session_grant(self, tool_name: str) -> bool:
        with self._lock:
            return tool_name in self._session_grants

    def check(self, request: PermissionRequest, *, matcher: RuleMatcher | None = None) -> bool:
        """Return True if the call may proceed, prompting if the policy says ``ask``."""
        decision = self.decision_for(request.tool_name, request.kind, request.arguments, matcher)
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
