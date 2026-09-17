"""Hooks: application callbacks that can steer a run, not just observe it.

The :class:`~kennel.events.EventBus` reports what happened; hooks decide what
happens. An embedding application registers them on the agent::

    async def guard(call, ctx):                      # before_tool
        if call.name == "shell" and "curl" in call.arguments["command"]:
            return Deny("network access is not allowed here")
        if call.name == "read":
            return Allow(updated_arguments={**call.arguments, "start_line": 1})

    def add_context(prompt, session):               # before_prompt
        return "Today is 2026-09-18."

    agent = Agent(".", hooks=Hooks(before_tool=[guard], before_prompt=[add_context]))

Callbacks may be sync or async. They run on the thread the provider used for the
tool call, so they must not touch objects bound to another event loop; the agent
only ever hands them plain data and the shared, thread-safe services.

Returning ``None`` from a hook means "no opinion". A :class:`~kennel.Deny` stops
the call and its message goes back to the model; an :class:`~kennel.Allow` may
rewrite the arguments (they are re-validated) or remember the approval for the
session. ``after_tool`` returns a :class:`~kennel.ToolResult` to replace the
result, or ``None`` to keep it. A hook that raises fails the tool call: a broken
policy is an error, not an absence of policy.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .permissions import Allow, Deny, PermissionManager
from .tools.base import Tool, ToolContext, ToolResult
from .workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .session import Session

__all__ = [
    "Allow",
    "BeforePromptHook",
    "BeforeToolHook",
    "Deny",
    "HookContext",
    "HookMatcher",
    "Hooks",
    "ToolCallRequest",
    "run_hook",
]


@dataclass(frozen=True)
class ToolCallRequest:
    """The call a ``before_tool`` / ``after_tool`` hook is looking at.

    ``arguments`` is the validated mapping the tool would run with; treat it as
    read-only and return ``Allow(updated_arguments=...)`` to change it.
    """

    name: str
    arguments: Mapping[str, Any]
    summary: str


@dataclass(frozen=True)
class HookContext:
    """What a hook may look at besides the call itself.

    It wraps the same :class:`~kennel.ToolContext` the tool will run with, so a
    hook sees exactly what the tool sees, and nothing has to be re-plumbed here
    when that context grows.
    """

    session_id: str
    tool_context: ToolContext
    tool: Tool | None = None

    @property
    def workspace(self) -> Workspace:
        return self.tool_context.workspace

    @property
    def permissions(self) -> PermissionManager:
        return self.tool_context.permission_manager

    @property
    def environment(self) -> Mapping[str, str]:
        return self.tool_context.environment


HookResult = Allow | Deny | None
BeforeToolHook = Callable[[ToolCallRequest, HookContext], HookResult | Awaitable[HookResult]]
AfterToolHook = Callable[[ToolCallRequest, ToolResult, HookContext], "ToolResult | None | Awaitable[ToolResult | None]"]
BeforePromptHook = Callable[[str, "Session"], str | None | Awaitable[str | None]]
ToolHook = BeforeToolHook | AfterToolHook


@dataclass
class HookMatcher:
    """Restrict hooks to some tools: ``HookMatcher(tools=["shell"], hooks=[guard])``."""

    tools: Sequence[str] | None = None
    hooks: Sequence[ToolHook] = ()

    def matches(self, tool_name: str) -> bool:
        return self.tools is None or tool_name in self.tools


@dataclass
class Hooks:
    """The callbacks an application registers. Entries may be added later.

    Each tool list holds plain callables and/or :class:`HookMatcher` entries;
    ``agent.hooks.before_tool.append(...)`` works after the agent is built.
    """

    before_tool: list[BeforeToolHook | HookMatcher] = field(default_factory=list)
    after_tool: list[AfterToolHook | HookMatcher] = field(default_factory=list)
    before_prompt: list[BeforePromptHook] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.before_tool or self.after_tool or self.before_prompt)


def select_hooks(entries: Sequence[ToolHook | HookMatcher], tool_name: str) -> list[ToolHook]:
    """Flatten ``entries`` into the callbacks that apply to ``tool_name``."""
    selected: list[ToolHook] = []
    for entry in entries:
        if isinstance(entry, HookMatcher):
            if entry.matches(tool_name):
                selected.extend(entry.hooks)
        else:
            selected.append(entry)
    return selected


async def run_hook(hook: Callable[..., Any], *args: Any) -> Any:
    """Call a hook that may be sync or async, and return its value."""
    result = hook(*args)
    if inspect.isawaitable(result):
        return await result
    return result
