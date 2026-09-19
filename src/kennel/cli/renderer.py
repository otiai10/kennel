"""Terminal rendering of agent events, streamed answers and permission prompts."""

from __future__ import annotations

import json
import select
import sys
import threading
from typing import IO

from ..events import Event, EventBus, EventType
from ..permissions import Approval, PermissionRequest


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


class Renderer:
    """Prints one line per tool call and streams the answer, keeping the two apart.

    With ``quiet`` nothing is written to stdout (``--output-format json`` owns it);
    ``--trace`` and ``error()`` still write to stderr.
    """

    def __init__(self, out: IO[str] = sys.stdout, err: IO[str] = sys.stderr, *, verbose: bool = False, trace: bool = False, color: bool = True, quiet: bool = False) -> None:
        self.out = out
        self.err = err
        self.verbose = verbose
        self.trace = trace
        self.quiet = quiet
        self.style = Style(color)
        self._state = "idle"  # idle | tools | text
        self._lock = threading.Lock()
        self._last_char = "\n"

    def attach(self, events: EventBus) -> None:
        events.subscribe(self.on_event)

    # -- event consumer ----------------------------------------------------------

    def on_event(self, event: Event) -> None:
        if self.trace:
            self.err.write(json.dumps({"t": round(event.timestamp, 3), "type": event.type, **event.data}, ensure_ascii=False, default=str) + "\n")
            self.err.flush()
        if self.quiet:
            return
        s = self.style
        t = event.type
        d = event.data
        if t == EventType.TOOL_STARTED:
            self._line(f"{s.cyan('●')} {d.get('summary', '')}")
        elif t == EventType.TOOL_COMPLETED and self.verbose:
            extra = ", truncated" if d.get("truncated") else ""
            self._line(s.dim(f"  ↳ {d.get('output_bytes', 0)} bytes{extra} in {d.get('duration_ms', 0):.0f} ms"))
        elif t == EventType.TOOL_FAILED:
            self._line(s.red(f"  ✗ {d.get('error', 'failed')}"))
        elif t == EventType.PERMISSION_DENIED:
            self._line(s.yellow(f"  ⊘ denied: {d.get('summary', '')}"))
        elif t == EventType.MODEL_NUDGED:
            rule = d.get("rule")
            suffix = f" (rule: {rule})" if rule else ""
            self._line(s.dim(f"  ↻ carrying out the described steps instead of narrating them{suffix}"))
        elif t == EventType.CONTEXT_COMPACTED:
            self._line(s.dim("  ↻ context compacted, retrying"))

    def _line(self, text: str) -> None:
        with self._lock:
            if self._state == "text" and self._last_char != "\n":
                self.out.write("\n")
            self.out.write(text + "\n")
            self.out.flush()
            self._state = "tools"
            self._last_char = "\n"

    # -- answer streaming --------------------------------------------------------

    def delta(self, text: str) -> None:
        if not text or self.quiet:
            return
        with self._lock:
            if self._state == "tools":
                self.out.write("\n")
            self._state = "text"
            self.out.write(text)
            self.out.flush()
            self._last_char = text[-1]

    def finish_answer(self) -> None:
        if self.quiet:
            return
        with self._lock:
            if self._state != "idle" and self._last_char != "\n":
                self.out.write("\n")
            self.out.flush()
            self._state = "idle"
            self._last_char = "\n"

    # -- misc ----------------------------------------------------------------------

    def note(self, text: str) -> None:
        if self.quiet:
            return
        self.finish_answer()
        self.out.write(self.style.dim(text) + "\n")
        self.out.flush()

    def error(self, text: str) -> None:
        self.finish_answer()
        self.err.write(self.style.red(f"error: {text}") + "\n")
        self.err.flush()


class ConsolePrompter:
    """Interactive approval prompt. Safe to call from a worker thread; cancellable."""

    def __init__(self, out: IO[str] = sys.stdout, inp: IO[str] = sys.stdin, *, color: bool = True, max_detail_lines: int = 40) -> None:
        self.out = out
        self.inp = inp
        self.style = Style(color)
        self.max_detail_lines = max_detail_lines
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def reset(self) -> None:
        self._cancelled.clear()

    def __call__(self, request: PermissionRequest) -> Approval:
        s = self.style
        lines = [f"\n{s.bold(s.yellow('?'))} Allow {request.kind.value.capitalize()}: {s.bold(request.summary)}"]
        if request.details:
            detail_lines = request.details.rstrip("\n").splitlines()
            shown = detail_lines[: self.max_detail_lines]
            lines += ["  " + s.dim(line) for line in shown]
            if len(detail_lines) > len(shown):
                lines.append(s.dim(f"  ... ({len(detail_lines) - len(shown)} more lines)"))
        for warning in request.warnings:
            lines.append(s.red(f"  ! {warning}"))
        self.out.write("\n".join(lines) + "\n")
        while True:
            self.out.write("  [y] once  [a] session  [n] deny > ")
            self.out.flush()
            answer = self._read_line()
            if answer is None:
                if not self._cancelled.is_set():  # EOF; on cancel the caller reports instead
                    self.out.write(s.dim("  denied\n"))
                    self.out.flush()
                return Approval.DENY
            answer = answer.strip().lower()
            if answer in ("y", "yes", "once"):
                return Approval.ONCE
            if answer in ("a", "always", "session"):
                return Approval.SESSION
            if answer in ("n", "no", "deny", ""):
                return Approval.DENY
            self.out.write(s.dim("  please answer y, a or n\n"))

    def _read_line(self) -> str | None:
        """Read one line, polling so a cancel (Ctrl-C elsewhere) or EOF ends the prompt."""
        while not self._cancelled.is_set():
            try:
                ready, _, _ = select.select([self.inp], [], [], 0.1)
            except (OSError, ValueError):
                line = self.inp.readline()
                return line if line else None
            if ready:
                line = self.inp.readline()
                return line if line else None
        return None
