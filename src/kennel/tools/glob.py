"""glob: find files by pattern inside the workspace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import ToolExecutionError
from ..permissions import PermissionKind
from ..rules import match_path_argument, matches_glob
from ._fs import iter_files
from .base import Tool, ToolContext, ToolParameter, ToolResult


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files in the workspace by glob pattern, e.g. '**/*.md' or 'transcripts/*.txt'. "
        "A pattern without '/' matches file names at any depth. Returns relative paths, "
        "one per line. Use this to discover files before reading them."
    )
    permission = PermissionKind.READ
    parameters = (
        ToolParameter("pattern", "string", "Glob pattern such as **/*.py"),
        ToolParameter("path", "string", "Directory to search, relative to the workspace (default: .)", required=False),
        ToolParameter("include_hidden", "boolean", "Include dot-files and dot-directories (default: false)", required=False),
    )

    def summarize(self, arguments: dict[str, Any]) -> str:
        path = arguments.get("path")
        suffix = f" in {path}" if path and path not in (".", "./") else ""
        return f"Glob {arguments.get('pattern', '')}{suffix}"

    def match_rule(self, specifier: str, arguments: Mapping[str, Any]) -> bool:
        """``glob(src/**)``: the specifier is a glob over the search root (default ``.``)."""
        return match_path_argument(specifier, arguments, default=".")

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = arguments["pattern"].strip()
        base = context.workspace.resolve(arguments.get("path") or ".")
        if not base.exists():
            raise ToolExecutionError(f"No such directory: {context.workspace.relative(base)}")
        limit = context.limits.glob_max_results
        include_hidden = bool(arguments.get("include_hidden", False))
        if pattern.startswith("./"):
            pattern = pattern[2:]
        matched: list[str] = []
        total = 0
        for file in iter_files(context.workspace, base, include_hidden=include_hidden):
            rel_to_base = file.relative_to(base).as_posix() if base.is_dir() else file.name
            if matches_glob(pattern, rel_to_base):
                total += 1
                if len(matched) < limit:
                    matched.append(context.workspace.relative(file))
        truncated = total > len(matched)
        if not matched:
            content = f"No files matched {pattern!r}.\n" + _overview(context, base)
        else:
            content = "\n".join(matched)
            if truncated:
                content += f"\n... {total - len(matched)} more matches not shown (narrow the pattern)"
        return ToolResult(
            content=content,
            metadata={"total": total, "returned": len(matched), "pattern": pattern},
            truncated=truncated,
        )


def _overview(context: ToolContext, base) -> str:
    """Short listing of the top-level entries so the model can broaden its search."""
    ws = context.workspace
    entries: list[str] = []
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return ""
    for child in children:
        if child.name.startswith(".") or ws.is_ignored_dir(child.name):
            continue
        if child.is_dir() and not child.is_symlink():
            count = sum(1 for _ in iter_files(ws, child))
            entries.append(f"{ws.relative(child)}/ ({count} files)")
        elif child.is_file():
            entries.append(ws.relative(child))
        if len(entries) >= 20:
            entries.append("...")
            break
    if not entries:
        return f"{ws.relative(base)} is empty."
    return "Top-level entries here: " + ", ".join(entries)
