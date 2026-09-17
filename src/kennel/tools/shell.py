"""shell: run a command in the workspace (requires permission).

The command string is shown verbatim for approval and executed with ``/bin/sh -c``
in the workspace directory, without stdin, with a timeout, a bounded output and
an allow-listed environment. Dangerous-looking commands add a warning to the
approval prompt; that heuristic is not a security boundary.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
from typing import Any

from ..errors import ToolArgumentError
from ..permissions import PermissionKind
from .base import Tool, ToolContext, ToolParameter, ToolResult

ENV_ALLOWLIST = ("PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ")

DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b", "recursive force delete (rm -rf)"),
    (r"\bsudo\b", "runs with sudo"),
    (r"\b(mkfs|diskutil\s+(erase|partition)|dd\s+if=)", "disk formatting / raw disk write"),
    (r"\b(shutdown|reboot|halt)\b", "shuts down or reboots the machine"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f|push\s+.*--force)", "destructive git operation"),
    (r"(^|[\s;&|])(/|~)[^\s]*\s*$|>\s*(/|~)", "references paths outside the workspace"),
    (r"\bcurl\b.*\|\s*(sh|bash|zsh)\b", "pipes a download into a shell"),
)


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a shell command in the workspace directory and return its exit code, stdout and "
        "stderr. Requires user approval. Commands must be non-interactive and finish within "
        "the timeout; output is truncated."
    )
    permission = PermissionKind.SHELL
    parameters = (
        ToolParameter("command", "string", "The command line to run with /bin/sh -c"),
        ToolParameter("timeout_seconds", "integer", "Timeout in seconds (default: 30)", required=False),
    )

    def summarize(self, arguments: dict[str, Any]) -> str:
        return f"Shell {arguments.get('command', '')}"

    def permission_details(self, arguments: dict[str, Any], context: ToolContext) -> str | None:
        return f"$ {arguments['command']}\n(cwd: {context.workspace.root})"

    def permission_warnings(self, arguments: dict[str, Any], context: ToolContext) -> tuple[str, ...]:
        return tuple(reason for pattern, reason in DANGEROUS_PATTERNS if re.search(pattern, arguments["command"]))

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        command = arguments["command"]
        if not command.strip():
            raise ToolArgumentError("shell: command must not be empty")
        timeout = int(arguments.get("timeout_seconds") or context.limits.shell_timeout_seconds)
        timeout = max(1, min(timeout, 600))
        env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
        env.update(context.environment)
        max_bytes = context.limits.max_output_bytes
        return await asyncio.to_thread(_run, command, str(context.workspace.root), timeout, env, max_bytes)


def _run(command: str, cwd: str, timeout: int, env: dict[str, str], max_bytes: int) -> ToolResult:
    proc = subprocess.Popen(
        ["/bin/sh", "-c", command],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
    half = max(1024, max_bytes // 2)
    stdout, t1 = _bounded(out, half)
    stderr, t2 = _bounded(err, half)
    status = f"exit code: {proc.returncode}"
    if timed_out:
        status += f" (killed after {timeout}s timeout)"
    parts = [status]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return ToolResult(
        content="\n".join(parts),
        metadata={"exit_code": proc.returncode, "timed_out": timed_out, "stdout_bytes": len(out), "stderr_bytes": len(err)},
        truncated=t1 or t2,
    )


def _bounded(data: bytes, limit: int) -> tuple[str, bool]:
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace").rstrip("\n"), False
    text = data[:limit].decode("utf-8", errors="replace")
    return text + f"\n... [truncated {len(data) - limit} bytes]", True
