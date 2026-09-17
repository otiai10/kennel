"""write: create a file or replace its whole content (requires permission)."""

from __future__ import annotations

from typing import Any

from ..errors import ToolExecutionError
from ..permissions import PermissionKind
from ._fs import atomic_write_text, is_probably_binary, unified_diff
from .base import Tool, ToolContext, ToolParameter, ToolResult


class WriteTool(Tool):
    name = "write"
    description = (
        "Create a new file or replace the entire content of an existing file in the "
        "workspace. Requires user approval. For small changes to an existing file prefer edit."
    )
    permission = PermissionKind.WRITE
    parameters = (
        ToolParameter("path", "string", "File path relative to the workspace"),
        ToolParameter("content", "string", "Full content to write"),
    )

    def summarize(self, arguments: dict[str, Any]) -> str:
        return f"Write {arguments.get('path', '')}"

    def permission_details(self, arguments: dict[str, Any], context: ToolContext) -> str | None:
        path = context.workspace.resolve(arguments["path"])
        rel = context.workspace.relative(path)
        content = arguments["content"]
        if path.exists():
            if path.is_dir():
                return f"{rel} is a directory (write will fail)"
            if is_probably_binary(path):
                return f"Overwrites existing binary file {rel} ({path.stat().st_size} -> {len(content.encode())} bytes)"
            old = path.read_text(encoding="utf-8", errors="replace")
            return f"Overwrites existing file {rel} ({len(old.encode())} -> {len(content.encode())} bytes)\n" + unified_diff(old, content, rel)
        return f"Creates new file {rel} ({len(content.encode())} bytes, {content.count(chr(10)) + (0 if content.endswith(chr(10)) or not content else 1)} lines)"

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        ws = context.workspace
        path = ws.resolve(arguments["path"])
        rel = ws.relative(path)
        if path.is_dir():
            raise ToolExecutionError(f"{rel} is a directory")
        existed = path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            n = atomic_write_text(path, arguments["content"])
        except OSError as exc:
            raise ToolExecutionError(f"write failed for {rel}: {exc}") from exc
        return ToolResult(
            content=f"{'Overwrote' if existed else 'Created'} {rel} ({n} bytes)",
            metadata={"path": rel, "bytes": n, "overwritten": existed},
        )
