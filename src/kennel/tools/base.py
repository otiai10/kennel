"""Tool interface shared by built-in and custom tools."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..errors import ToolArgumentError
from ..permissions import PermissionKind, PermissionManager
from ..rules import match_arguments
from ..workspace import Workspace

ParamType = Literal["string", "integer", "number", "boolean"]


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: ParamType
    description: str
    required: bool = True


@dataclass
class ToolLimits:
    """Size/time limits applied by tools. Values are tuned by measurement."""

    max_output_bytes: int = 64 * 1024
    max_read_lines: int = 400
    max_file_bytes: int = 2_000_000
    glob_max_results: int = 500
    grep_max_results: int = 100
    grep_snippet_chars: int = 240
    shell_timeout_seconds: int = 30


@dataclass
class ToolContext:
    workspace: Workspace
    permission_manager: PermissionManager
    environment: Mapping[str, str] = field(default_factory=dict)
    limits: ToolLimits = field(default_factory=ToolLimits)


@dataclass
class ToolResult:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False


class Tool(ABC):
    """A capability the model can invoke.

    Subclasses set ``name``, ``description``, ``permission`` and ``parameters``
    and implement :meth:`execute`. Arguments arrive validated and coerced
    according to ``parameters``.
    """

    name: str
    description: str
    permission: PermissionKind = PermissionKind.READ
    parameters: Sequence[ToolParameter] = ()

    @abstractmethod
    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """Run the tool. Raise a :class:`~kennel.errors.ToolError` subclass on failure."""

    # -- presentation hooks (used by the CLI and permission prompts) -------

    def summarize(self, arguments: dict[str, Any]) -> str:
        """One-line, human-readable description of a call, e.g. ``Glob **/*.py``."""
        return f"{self.name.capitalize()} {json.dumps(arguments, ensure_ascii=False)}"

    def permission_details(self, arguments: dict[str, Any], context: ToolContext) -> str | None:
        """Extra text shown in the approval prompt (a diff, a command, ...)."""
        return None

    def permission_warnings(self, arguments: dict[str, Any], context: ToolContext) -> tuple[str, ...]:
        return ()

    # -- permission rules -----------------------------------------------------

    def match_rule(self, specifier: str, arguments: Mapping[str, Any]) -> bool:
        """Does the ``specifier`` of a rule ``name(specifier)`` cover this call?

        The tool owns the meaning of its specifiers: ``shell`` matches the
        command line, the file tools match the path. The default joins the
        argument values and matches them as text, which is what a custom tool
        gets for free. Called from the provider's tool thread, so it must not
        touch mutable state.
        """
        return match_arguments(specifier, arguments)

    # -- validation -----------------------------------------------------------

    def validate(self, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        """Validate and coerce raw model arguments. Unknown keys are dropped."""
        raw = dict(arguments or {})
        out: dict[str, Any] = {}
        for param in self.parameters:
            if param.name not in raw or raw[param.name] is None:
                if param.required:
                    raise ToolArgumentError(f"{self.name}: missing required argument '{param.name}'")
                continue
            out[param.name] = _coerce(self.name, param, raw[param.name])
        return out


def _coerce(tool_name: str, param: ToolParameter, value: Any) -> Any:
    kind = param.type
    try:
        if kind == "string":
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float, bool)):
                return str(value)
        elif kind == "integer":
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                return int(value.strip())
        elif kind == "number":
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                return float(value.strip())
        elif kind == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
    except ValueError:
        pass
    raise ToolArgumentError(
        f"{tool_name}: argument '{param.name}' must be {kind}, got {type(value).__name__}: {value!r}"
    )
