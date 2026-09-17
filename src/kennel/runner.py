"""ToolRunner: the single path every model-requested tool call goes through.

Validation, guardrails, permission, execution, output bounding and events all
happen here, so the agent loop stays provider-driven and the Agent class holds
no tool logic. Errors are returned to the model as short ``Error: ...`` strings
so it can adapt; they are also recorded for the trace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .context import truncate_text
from .errors import KennelError, ToolArgumentError
from .events import EventBus, EventType
from .permissions import Decision, PermissionRequest
from .tools.base import Tool, ToolContext

log = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    summary: str
    status: str  # ok | error | denied | blocked | invalid
    output_bytes: int = 0
    truncated: bool = False
    duration_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRunner:
    def __init__(
        self,
        tools: Mapping[str, Tool],
        context: ToolContext,
        events: EventBus,
        session_id: str,
        *,
        max_tool_calls: int = 32,
        max_repeats: int = 3,
    ) -> None:
        self._tools = dict(tools)
        self._context = context
        self._events = events
        self._session_id = session_id
        self.max_tool_calls = max_tool_calls
        self.max_repeats = max_repeats
        self._lock = threading.Lock()
        self.records: list[ToolCallRecord] = []
        self._turn_records: list[ToolCallRecord] = []
        self._turn_calls = 0
        self._recent: deque[tuple[str, str]] = deque(maxlen=8)
        self.limit_hit = False
        self._turn_cancelled = False

    def begin_turn(self) -> None:
        with self._lock:
            self._turn_records = []
            self._turn_calls = 0
            self._recent.clear()
            self.limit_hit = False
            self._turn_cancelled = False

    def cancel_turn(self) -> None:
        """Mark the current turn cancelled: in-flight calls are still recorded but emit no events."""
        with self._lock:
            self._turn_cancelled = True

    def turn_records(self) -> list[ToolCallRecord]:
        with self._lock:
            return list(self._turn_records)

    def _record(self, record: ToolCallRecord) -> ToolCallRecord:
        with self._lock:
            self.records.append(record)
            self._turn_records.append(record)
        return record

    def _emit(self, type: str, **data: Any) -> None:
        if self._turn_cancelled:
            return
        self._events.emit(type, self._session_id, **data)

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        summary = tool.summarize(arguments) if tool else f"{name} {json.dumps(arguments, ensure_ascii=False)}"
        self._emit(EventType.TOOL_REQUESTED, tool=name, summary=summary)

        if tool is None:
            error = f"unknown tool '{name}'"
            self._record(ToolCallRecord(name, dict(arguments), summary, "invalid", error=error))
            self._emit(EventType.TOOL_FAILED, tool=name, summary=summary, error=error)
            return f"Error: unknown tool '{name}'. Available tools: {', '.join(sorted(self._tools))}."

        try:
            args = tool.validate(arguments)
        except ToolArgumentError as exc:
            self._record(ToolCallRecord(name, dict(arguments), summary, "invalid", error=str(exc)))
            self._emit(EventType.TOOL_FAILED, tool=name, summary=summary, error=str(exc))
            return f"Error: {exc}"
        summary = tool.summarize(args)

        with self._lock:
            self._turn_calls += 1
            key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            self._recent.append(key)
            repeats = sum(1 for k in self._recent if k == key)
            over_limit = self._turn_calls > self.max_tool_calls
            if over_limit:
                self.limit_hit = True
        if over_limit:
            msg = f"tool call limit ({self.max_tool_calls} per turn) reached"
            self._record(ToolCallRecord(name, args, summary, "blocked", error=msg))
            self._emit(EventType.TOOL_FAILED, tool=name, summary=summary, error=msg)
            return "Error: the tool call limit for this turn was reached. Stop calling tools and answer with what you have."
        if repeats >= self.max_repeats:
            msg = "repeated identical tool call"
            self._record(ToolCallRecord(name, args, summary, "blocked", error=msg))
            self._emit(EventType.TOOL_FAILED, tool=name, summary=summary, error=msg)
            return "Error: this exact tool call was already made with the same arguments. Do not repeat it; use the earlier result or change the arguments."

        pm = self._context.permission_manager
        if pm.decision_for(name, tool.permission, args, tool.match_rule) is not Decision.ALLOW:
            try:
                details = tool.permission_details(args, self._context)
                warnings = tool.permission_warnings(args, self._context)
            except KennelError as exc:
                return self._fail(name, args, summary, exc)
            request = PermissionRequest(name, tool.permission, summary, details, warnings, dict(args))
            self._emit(EventType.PERMISSION_REQUESTED, tool=name, summary=summary, warnings=list(warnings))
            if not pm.check(request, matcher=tool.match_rule):
                self._record(ToolCallRecord(name, args, summary, "denied", error="permission denied"))
                self._emit(EventType.PERMISSION_DENIED, tool=name, summary=summary)
                return f"Error: permission denied for {name}. The user did not approve this action; do not retry it, explain what you would have done instead."

        self._emit(EventType.TOOL_STARTED, tool=name, summary=summary)
        started = time.perf_counter()
        try:
            result = await tool.execute(args, self._context)
        except asyncio.CancelledError:
            raise
        except KennelError as exc:  # tool errors and workspace violations
            return self._fail(name, args, summary, exc, started)
        except Exception as exc:  # noqa: BLE001 - tool bugs must not kill the session
            log.exception("tool %s crashed", name)
            return self._fail(name, args, summary, exc, started, prefix=f"{name} failed: ")

        content, truncated = truncate_text(result.content, self._context.limits.max_output_bytes)
        truncated = truncated or result.truncated
        duration = (time.perf_counter() - started) * 1000
        record = self._record(
            ToolCallRecord(name, args, summary, "ok", len(content.encode()), truncated, duration, None, dict(result.metadata))
        )
        self._emit(
            EventType.TOOL_COMPLETED,
            tool=name,
            summary=summary,
            output_bytes=record.output_bytes,
            truncated=truncated,
            duration_ms=duration,
            metadata=dict(result.metadata),
        )
        return content

    def _fail(self, name: str, args: dict[str, Any], summary: str, exc: BaseException, started: float | None = None, prefix: str = "") -> str:
        duration = (time.perf_counter() - started) * 1000 if started else 0.0
        message = f"{prefix}{exc}"
        self._record(ToolCallRecord(name, args, summary, "error", duration_ms=duration, error=message))
        self._emit(EventType.TOOL_FAILED, tool=name, summary=summary, error=message, duration_ms=duration)
        return f"Error: {message}"
