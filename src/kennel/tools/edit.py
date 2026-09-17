"""edit: exact, unique text replacement in an existing file (requires permission)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import ToolArgumentError, ToolExecutionError
from ..permissions import PermissionKind
from ..rules import match_path_argument
from ._fs import atomic_write_text, is_probably_binary, unified_diff
from .base import Tool, ToolContext, ToolParameter, ToolResult


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace one exact occurrence of old_text with new_text in an existing file. "
        "old_text must appear exactly once; include enough surrounding lines to make it unique. "
        "Requires user approval."
    )
    permission = PermissionKind.WRITE
    parameters = (
        ToolParameter("path", "string", "File path relative to the workspace"),
        ToolParameter("old_text", "string", "Exact text to replace (must be unique in the file)"),
        ToolParameter("new_text", "string", "Replacement text"),
    )

    def summarize(self, arguments: dict[str, Any]) -> str:
        return f"Edit {arguments.get('path', '')}"

    def _prepare(self, arguments: dict[str, Any], context: ToolContext) -> tuple[str, str, str]:
        ws = context.workspace
        path = ws.resolve(arguments["path"])
        rel = ws.relative(path)
        old_text, new_text = arguments["old_text"], arguments["new_text"]
        if not old_text:
            raise ToolArgumentError("edit: old_text must not be empty")
        if not path.exists():
            raise ToolExecutionError(f"No such file: {rel} (use write to create files)")
        if path.is_dir() or is_probably_binary(path):
            raise ToolExecutionError(f"{rel} is not a text file")
        original = path.read_text(encoding="utf-8", errors="replace")
        count = original.count(old_text)
        if count == 0:
            raise ToolExecutionError(f"edit: old_text was not found in {rel}; read the file and copy the text exactly")
        if count > 1:
            raise ToolExecutionError(f"edit: old_text is ambiguous ({count} matches in {rel}); include more context to make it unique")
        updated = original.replace(old_text, new_text, 1)
        return rel, original, updated

    def permission_details(self, arguments: dict[str, Any], context: ToolContext) -> str | None:
        rel, original, updated = self._prepare(arguments, context)
        return unified_diff(original, updated, rel)

    def match_rule(self, specifier: str, arguments: Mapping[str, Any]) -> bool:
        """``edit(docs/**)``: the specifier is a glob over the workspace-relative path."""
        return match_path_argument(specifier, arguments)

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        rel, original, updated = self._prepare(arguments, context)
        path = context.workspace.resolve(arguments["path"])
        try:
            n = atomic_write_text(path, updated)
        except OSError as exc:
            raise ToolExecutionError(f"edit failed for {rel}: {exc}") from exc
        delta = updated.count("\n") - original.count("\n")
        return ToolResult(
            content=f"Edited {rel} (1 replacement, {delta:+d} lines)",
            metadata={"path": rel, "bytes": n, "line_delta": delta},
        )
