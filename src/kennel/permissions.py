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

#: Tool names a read-only agent is limited to. ``registry.READ_ONLY_TOOLS`` is this
#: tuple; it lives here because :class:`PermissionMode` needs it and ``registry``
#: imports this module (not the other way round).
READ_ONLY_TOOL_NAMES: tuple[str, ...] = ("glob", "grep", "read")

_A, _Q, _D = Decision.ALLOW, Decision.ASK, Decision.DENY

_DEFAULT_KIND_DEFAULTS: dict[PermissionKind, Decision] = {
    PermissionKind.READ: _A,
    PermissionKind.WRITE: _Q,
    PermissionKind.SHELL: _Q,
    PermissionKind.WEB: _D,
}

#: ``dont-ask`` and ``read-only`` are ``default`` with every ``ask`` turned into ``deny``.
_NO_ASK_KIND_DEFAULTS: dict[PermissionKind, Decision] = {
    kind: (_D if decision is _Q else decision) for kind, decision in _DEFAULT_KIND_DEFAULTS.items()
}

_MODE_KIND_DEFAULTS: dict[str, dict[PermissionKind, Decision]] = {
    "read-only": _NO_ASK_KIND_DEFAULTS,
    "default": _DEFAULT_KIND_DEFAULTS,
    "accept-edits": {**_DEFAULT_KIND_DEFAULTS, PermissionKind.WRITE: _A, PermissionKind.WEB: _Q},
    "dont-ask": _NO_ASK_KIND_DEFAULTS,
    "bypass": dict.fromkeys(_DEFAULT_KIND_DEFAULTS, _A),
}


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
        return READ_ONLY_TOOL_NAMES if self is PermissionMode.READ_ONLY else None

    def kind_defaults(self) -> dict[PermissionKind, Decision]:
        """Default decision per permission kind, used for tools the policy does not name."""
        return dict(_MODE_KIND_DEFAULTS[self.value])

    def policy(self) -> dict[str, Decision]:
        """The mode expressed as a decision per built-in tool name."""
        defaults = self.kind_defaults()
        return {name: defaults[kind] for name, kind in BUILTIN_KINDS.items()}


DEFAULT_MODE = PermissionMode.DEFAULT


@dataclass(frozen=True)
class Allow:
    """Let the call proceed, optionally with corrected arguments.

    ``updated_arguments`` replaces the call's arguments (they are re-validated).
    ``remember="session"`` grants the tool for the rest of the session, like
    :attr:`Approval.SESSION` (within the tool's :meth:`~kennel.Tool.session_scope`).
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
    # Validated tool arguments, so rule specifiers and application hooks can see
    # what is actually being asked for. Excluded from equality so the request
    # stays hashable.
    arguments: Mapping[str, Any] = field(default_factory=dict, compare=False)
    # What an Approval.SESSION answer grants (Tool.session_scope): None for the whole
    # tool, or e.g. the host for fetch, so a prompter can say what "session" covers.
    scope: str | None = None


@dataclass(frozen=True)
class PermissionRule:
    """One parsed policy entry: ``shell(git *) -> allow``."""

    tool_name: str
    specifier: str | None
    decision: Decision

    @property
    def key(self) -> str:
        return self.tool_name if self.specifier is None else f"{self.tool_name}({self.specifier})"


@dataclass(frozen=True)
class PermissionOutcome:
    """The resolved answer for one call, richer than a bool."""

    allowed: bool
    message: str | None = None
    updated_arguments: Mapping[str, Any] | None = None


Prompter = Callable[[PermissionRequest], "Approval | Allow | Deny"]
#: Decides whether a rule specifier covers a call; normally ``Tool.match_rule``.
RuleMatcher = Callable[[str, Mapping[str, Any]], bool]


def parse_rules(values: Mapping[str, Decision | str] | None) -> list[PermissionRule]:
    """Parse and validate a policy mapping into rules, preserving order.

    Both sides are checked: the key must be ``tool`` or ``tool(specifier)`` and
    the value must be a :class:`Decision`.
    """
    rules: list[PermissionRule] = []
    for key, value in (values or {}).items():
        tool_name, specifier = parse_rule(key)
        try:
            decision = value if isinstance(value, Decision) else Decision(str(value).lower())
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid permission {value!r} for rule {key!r}; expected allow, ask or deny"
            ) from exc
        rules.append(PermissionRule(tool_name, specifier, decision))
    return rules


def parse_policy(values: Mapping[str, Decision | str] | None) -> dict[str, Decision]:
    """Validate a policy mapping and return it keyed by canonical rule key."""
    return {rule.key: rule.decision for rule in parse_rules(values)}


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
        # One ordered rule list per tool, bare rules included (specifier None), so
        # resolving a call is a single dict lookup on the tool call hot path.
        self._rules: dict[str, list[PermissionRule]] = {}
        self._prompter = prompter
        # (tool name, scope); scope None grants the whole tool (Tool.session_scope).
        self._session_grants: set[tuple[str, str | None]] = set()
        self._lock = threading.Lock()
        self.update(policy)

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
        """Effective decision per tool name, ignoring specifier rules (for display)."""
        bare = {r.tool_name: r.decision for r in self.rules() if r.specifier is None}
        return {**self._defaults, **bare}

    def rules(self, *, specified_only: bool = False) -> list[PermissionRule]:
        """Declared rules, grouped per tool in declaration order."""
        with self._lock:
            return [
                rule
                for rules in self._rules.values()
                for rule in rules
                if not (specified_only and rule.specifier is None)
            ]

    def update(self, policy: Mapping[str, Decision | str] | None) -> None:
        """Layer more rules on top of the current policy (later wins)."""
        for rule in parse_rules(policy):
            with self._lock:
                kept = [r for r in self._rules.get(rule.tool_name, ()) if r.key != rule.key]
                self._rules[rule.tool_name] = kept + [rule]

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
            candidates = list(self._rules.get(tool_name, ()))
            kind_default = self._kind_defaults[kind]
        bare: Decision | None = None
        matched: set[Decision] = set()
        match = matcher or match_arguments
        for rule in candidates:
            if rule.specifier is None:
                bare = rule.decision
            elif arguments is not None and match(rule.specifier, arguments):
                matched.add(rule.decision)
        if Decision.DENY in matched or bare is Decision.DENY:
            return Decision.DENY
        if matched:
            return Decision.ASK if Decision.ASK in matched else Decision.ALLOW
        return bare if bare is not None else kind_default

    def set_decision(self, tool_name: str, decision: Decision | str) -> None:
        self.update({tool_name: decision})

    def grant_session(self, tool_name: str, scope: str | None = None) -> None:
        """Remember an approval for the rest of the session: the whole tool, or one ``scope``."""
        with self._lock:
            self._session_grants.add((tool_name, scope))

    def has_session_grant(self, tool_name: str, scope: str | None = None) -> bool:
        """Is this call covered: by a grant for the whole tool, or for exactly ``scope``?"""
        with self._lock:
            return (tool_name, None) in self._session_grants or (
                scope is not None and (tool_name, scope) in self._session_grants
            )

    def session_scopes(self, tool_name: str) -> list[str | None]:
        """The grants held for ``tool_name`` (``None`` = the whole tool), for display."""
        with self._lock:
            return sorted((s for t, s in self._session_grants if t == tool_name), key=lambda s: s or "")

    def check(
        self,
        request: PermissionRequest,
        *,
        matcher: RuleMatcher | None = None,
        decision: Decision | None = None,
    ) -> bool:
        """Return True if the call may proceed, prompting if the policy says ``ask``."""
        return self.decide(request, matcher=matcher, decision=decision).allowed

    def decide(
        self,
        request: PermissionRequest,
        *,
        matcher: RuleMatcher | None = None,
        decision: Decision | None = None,
    ) -> PermissionOutcome:
        """Resolve one call, prompting if the policy says ``ask``.

        A caller that already resolved the decision (the tool runner does, to
        decide whether to ask at all) passes it in so rules are matched once.
        The prompter may answer with an :class:`Approval`, or with
        :class:`Allow` / :class:`Deny` to add a reason or corrected arguments;
        both reach the caller through :class:`PermissionOutcome`.
        """
        if decision is None:
            decision = self.decision_for(request.tool_name, request.kind, request.arguments, matcher)
        if decision is Decision.ALLOW:
            return PermissionOutcome(True)
        if decision is Decision.DENY:
            return PermissionOutcome(False)
        if self.has_session_grant(request.tool_name, request.scope):
            return PermissionOutcome(True)
        if self._prompter is None:
            return PermissionOutcome(False)  # non-interactive: ask => deny
        return self.resolve(request.tool_name, self._prompter(request), scope=request.scope)

    def resolve(
        self, tool_name: str, answer: Approval | Allow | Deny | None, *, scope: str | None = None
    ) -> PermissionOutcome:
        """Interpret one answer about ``tool_name``, wherever it came from.

        Prompters and application hooks speak the same vocabulary, so what an
        ``Allow`` or a ``Deny`` *means* — including remembering a grant for the
        session, within ``scope`` (the tool's :meth:`~kennel.Tool.session_scope`)
        — is decided here and nowhere else. ``None`` means "no opinion"
        and leaves the call allowed to continue.
        """
        if isinstance(answer, Deny):
            return PermissionOutcome(False, message=answer.message)
        if isinstance(answer, Allow):
            if answer.remember_session:
                self.grant_session(tool_name, scope)
            return PermissionOutcome(True, updated_arguments=answer.updated_arguments)
        if answer is Approval.SESSION:
            self.grant_session(tool_name, scope)
            return PermissionOutcome(True)
        if answer is None:
            return PermissionOutcome(True)
        return PermissionOutcome(answer is Approval.ONCE)
