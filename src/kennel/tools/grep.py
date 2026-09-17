"""grep: search file contents; uses ripgrep when available, otherwise a Python fallback."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..errors import ToolArgumentError, ToolExecutionError
from ..permissions import PermissionKind
from ..rules import match_path_argument
from ._fs import is_probably_binary, iter_files, matches_glob
from .base import Tool, ToolContext, ToolParameter, ToolResult


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents in the workspace with a regular expression. Returns "
        "'path:line: text' matches (bounded). Prefer grep to locate relevant lines, then "
        "read only the needed line range."
    )
    permission = PermissionKind.READ
    parameters = (
        ToolParameter("pattern", "string", "Regular expression to search for (must not be empty; use read to view a whole file)"),
        ToolParameter("path", "string", "File or directory to search, relative to the workspace (default: .)", required=False),
        ToolParameter("glob", "string", "Only search files matching this glob, e.g. *.py", required=False),
        ToolParameter("case_sensitive", "boolean", "Case-sensitive search (default: false)", required=False),
        ToolParameter("max_results", "integer", "Maximum number of matching lines to return", required=False),
    )

    def __init__(self, *, use_rg: bool | None = None) -> None:
        self._use_rg = use_rg

    def summarize(self, arguments: dict[str, Any]) -> str:
        parts = [f"Grep {arguments.get('pattern', '')!r}"]
        if arguments.get("glob"):
            parts.append(f"({arguments['glob']})")
        if arguments.get("path") and arguments["path"] not in (".", "./"):
            parts.append(f"in {arguments['path']}")
        return " ".join(parts)

    def match_rule(self, specifier: str, arguments: Mapping[str, Any]) -> bool:
        """``grep(src/**)``: the specifier is a glob over the search root (default ``.``)."""
        return match_path_argument(specifier, arguments, default=".")

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = arguments["pattern"]
        if not pattern:
            raise ToolArgumentError("grep: pattern must not be empty")
        base = context.workspace.resolve(arguments.get("path") or ".")
        if not base.exists():
            raise ToolExecutionError(f"No such path: {context.workspace.relative(base)}")
        limit = max(1, min(int(arguments.get("max_results") or context.limits.grep_max_results), context.limits.grep_max_results))
        case_sensitive = bool(arguments.get("case_sensitive", False))
        file_glob = arguments.get("glob") or None
        use_rg = self._use_rg if self._use_rg is not None else shutil.which("rg") is not None

        if use_rg:
            lines, total, truncated = self._rg(context, base, pattern, file_glob, case_sensitive, limit)
        else:
            lines, total, truncated = self._python(context, base, pattern, file_glob, case_sensitive, limit)

        if not lines:
            content = f"No matches for {pattern!r}."
        else:
            content = "\n".join(lines)
            if truncated:
                content += f"\n... more matches not shown (limit {limit}); narrow the pattern or path"
        return ToolResult(
            content=content,
            metadata={"returned": len(lines), "total_seen": total, "truncated": truncated, "engine": "rg" if use_rg else "python"},
            truncated=truncated,
        )

    # -- ripgrep --------------------------------------------------------------

    def _rg(self, context: ToolContext, base: Path, pattern: str, file_glob: str | None, case_sensitive: bool, limit: int):
        ws = context.workspace
        cmd = [
            "rg", "--no-heading", "--line-number", "--color", "never", "--no-messages",
            "--sort", "path", "--max-columns", str(context.limits.grep_snippet_chars),
            "--max-filesize", str(context.limits.max_file_bytes),
        ]
        cmd.append("--case-sensitive" if case_sensitive else "--ignore-case")
        for ignored in sorted(ws.ignored_dirs):
            cmd += ["--glob", f"!{ignored}"]
        if file_glob:
            cmd += ["--glob", file_glob]
        cmd += ["--regexp", pattern, "--", str(base)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ws.root, errors="replace")
        except OSError as exc:  # pragma: no cover - rg missing at runtime
            raise ToolExecutionError(f"grep: failed to run rg: {exc}") from exc
        lines: list[str] = []
        total = 0
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                total += 1
                if len(lines) >= limit:
                    break
                lines.append(self._format_rg_line(ws, base, raw.rstrip("\n"), context.limits.grep_snippet_chars))
        finally:
            if proc.poll() is None:
                proc.kill()
            _, stderr = proc.communicate()
        if proc.returncode == 2 and not lines:
            raise ToolArgumentError(f"grep: {stderr.strip().splitlines()[-1] if stderr.strip() else 'invalid pattern'}")
        return lines, total, total > len(lines)

    @staticmethod
    def _format_rg_line(ws, base: Path, raw: str, snippet: int) -> str:
        path_part, sep, rest = raw.partition(":")
        if not sep:
            return raw
        if base.is_file():
            # rg prints only "line:text" for a single file target
            return f"{ws.relative(base)}:{raw}"[: snippet + 64]
        p = Path(path_part)
        rel = ws.relative(p if p.is_absolute() else ws.root / p)
        lineno, sep2, text = rest.partition(":")
        return f"{rel}:{lineno}: {text.strip()[:snippet]}" if sep2 else f"{rel}:{rest}"

    # -- python fallback -------------------------------------------------------

    def _python(self, context: ToolContext, base: Path, pattern: str, file_glob: str | None, case_sensitive: bool, limit: int):
        ws = context.workspace
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            raise ToolArgumentError(f"grep: invalid regular expression: {exc}") from exc
        snippet = context.limits.grep_snippet_chars
        lines: list[str] = []
        total = 0
        for file in iter_files(ws, base):
            rel = ws.relative(file)
            if file_glob and not matches_glob(file_glob, rel):
                continue
            try:
                if file.stat().st_size > context.limits.max_file_bytes or is_probably_binary(file):
                    continue
                with open(file, encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if regex.search(line):
                            total += 1
                            if len(lines) < limit:
                                lines.append(f"{rel}:{lineno}: {line.strip()[:snippet]}")
                            elif total > limit:
                                return lines, total, True
            except OSError:
                continue
        return lines, total, total > len(lines)
