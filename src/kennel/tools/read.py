"""read: bounded, line-numbered reads of text files."""

from __future__ import annotations

from typing import Any

from ..errors import ToolArgumentError, ToolExecutionError
from ..permissions import PermissionKind
from ._fs import is_probably_binary
from .base import Tool, ToolContext, ToolParameter, ToolResult


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a text file from the workspace with line numbers. Reads are bounded: give "
        "start_line and end_line to read a specific range, and read large files in "
        "chunks. Use grep first to find the relevant lines instead of reading whole files."
    )
    permission = PermissionKind.READ
    parameters = (
        ToolParameter("path", "string", "File path relative to the workspace"),
        ToolParameter("start_line", "integer", "First line to read, 1-based (default: 1)", required=False),
        ToolParameter("end_line", "integer", "Last line to read, inclusive (default: start_line + limit)", required=False),
    )

    def summarize(self, arguments: dict[str, Any]) -> str:
        s, e = arguments.get("start_line"), arguments.get("end_line")
        rng = f" [{s or 1}-{e}]" if e else (f" [{s}-]" if s else "")
        return f"Read {arguments.get('path', '')}{rng}"

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        ws = context.workspace
        path = ws.resolve(arguments["path"])
        rel = ws.relative(path)
        if not path.exists():
            raise ToolExecutionError(f"No such file: {rel}")
        if path.is_dir():
            raise ToolExecutionError(f"{rel} is a directory; use glob to list files")
        if is_probably_binary(path):
            raise ToolExecutionError(f"{rel} looks like a binary file; only text files can be read")

        max_lines = context.limits.max_read_lines
        start_arg = arguments.get("start_line")
        start = 1 if start_arg is None else int(start_arg)
        if start < 1:
            raise ToolArgumentError("read: start_line must be >= 1")
        end_arg = arguments.get("end_line")
        end = int(end_arg) if end_arg is not None else start + max_lines - 1
        if end < start:
            raise ToolArgumentError("read: end_line must be >= start_line")
        clamped = False
        if end - start + 1 > max_lines:
            end = start + max_lines - 1
            clamped = True

        size = path.stat().st_size
        count_all = size <= context.limits.max_file_bytes
        out: list[str] = []
        total = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                total = lineno
                if lineno < start:
                    continue
                if lineno <= end:
                    out.append(f"{lineno:>6}| {line.rstrip(chr(10))}")
                elif not count_all:
                    total = -1  # unknown; stop scanning huge files
                    break
        if total == 0:
            return ToolResult(content=f"{rel} is empty.", metadata={"path": rel, "total_lines": 0})
        if start > total >= 0:
            raise ToolArgumentError(f"read: start_line {start} is past the end of {rel} ({total} lines)")
        last = start + len(out) - 1
        has_more = total < 0 or last < total
        content = "\n".join(out)
        if has_more:
            total_text = f"of {total}" if total >= 0 else "of a large file"
            content += f"\n[showing lines {start}-{last} {total_text}; call read again with start_line={last + 1} for more]"
        return ToolResult(
            content=content,
            metadata={"path": rel, "start_line": start, "end_line": last, "total_lines": total if total >= 0 else None, "bytes": size, "clamped": clamped},
            truncated=has_more,
        )
