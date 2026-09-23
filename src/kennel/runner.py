"""ToolRunner: the single path every model-requested tool call goes through.

Validation, guardrails, hooks, permission, execution, output bounding and events
all happen here, so the agent loop stays provider-driven and the Agent class
holds no tool logic. Errors are returned to the model as short ``Error: ...``
strings so it can adapt; they are also recorded for the trace. A result too big for the
provider's whole context window is one of those cases: it is measured, then withheld and
replaced by a short instruction to ask for a smaller part, because handing it over does not
shorten the conversation, it fails the request.

Adapting is what the model is asked to do, not what it can be relied on to do. When the
same call keeps arriving after being refused as a repeat, the runner gives up on it
(:attr:`ToolRunner.repeat_wedged`): that call will not run again this turn, and its answer
becomes the same terminal instruction the spent budget would have given. Otherwise one
repeated call can spend the whole per-turn budget and the user loses the turn (issue #58).
A call that changes its arguments is still run, because narrowing the request is exactly what
the model was asked to do -- on a provider that runs its own tool loop it is the only
recovery there is.

``before_tool`` hooks run after validation and before the permission check, so an
application policy can deny a call the policy would have allowed, or correct its
arguments before anyone judges them. ``after_tool`` hooks can replace the result
before it is bounded. Every call may arrive on a different thread with its own
event loop (the provider's tool callback), so hook lists are snapshotted per call
and nothing loop-bound is kept.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .context import estimate_tokens_from_bytes, truncate_text
from .errors import KennelError, ToolArgumentError
from .events import EventBus, EventType
from .hooks import HookContext, Hooks, ToolCallRequest, run_hook, select_hooks
from .permissions import Decision, PermissionOutcome, PermissionRequest
from .tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)

#: The one thing a model can still usefully do once no further tool call will run this
#: turn, so the per-turn budget and the repeated-call guardrail say it with the same words.
STOP_CALLING_TOOLS = "Stop calling tools and answer with what you have."

#: What the model is given in place of a result that is larger than its whole context
#: window. Handing such a result over does not truncate the conversation, it fails the
#: request (Apple answers status 255), so the model is told how to ask for less instead.
#: The last sentence is there because the models that ignore this notice ignore it by
#: sending the identical call again (issue #58): that refusal is now said up front, by the
#: notice itself, and not only by the guardrail that answers the repeat.
WINDOW_EXCEEDED_NOTICE = (
    "Error: the {tool} result is ~{tokens:,} tokens, larger than this model's "
    "{window:,}-token context window, so it was not added to the conversation. "
    "Ask for a smaller part: read a line range (start_line/end_line), grep for the "
    "words you need, or work through the file section by section. Do not send this same "
    "call again with the same arguments: it will be refused."
)

#: The answer to an identical call that has already been made this turn.
REPEATED_CALL_REFUSAL = (
    "Error: this exact tool call was already made with the same arguments. Do not repeat "
    "it; use the earlier result or change the arguments."
)

#: The answer to a call the repeated-call guardrail has given up on (see
#: :attr:`ToolRunner.repeat_wedged`). It is the terminal instruction the model would
#: otherwise only have received at the end of the per-turn budget.
TOOLS_STOPPED_NOTICE = (
    "Error: this tool call kept arriving after being refused, so it will not run again this "
    f"turn. {STOP_CALLING_TOOLS}"
)


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
    #: The result was measured but not handed to the model: it does not fit the window
    #: on its own, so the model got :data:`WINDOW_EXCEEDED_NOTICE` instead. The sizes above
    #: still describe the result the tool produced.
    withheld: bool = False


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
        max_repeat_refusals: int = 3,
        hooks: Hooks | None = None,
        context_window_tokens: int | None = None,
    ) -> None:
        """``context_window_tokens`` is the provider's declared window, used to say on
        ``tool.completed`` whether a single result is already too big for it. ``None`` (the
        provider declares no window) means the question cannot be answered, not that the
        answer is no.

        ``max_repeats`` and ``max_repeat_refusals`` count different things and are therefore
        two numbers. ``max_repeats`` is how often the same call may appear among the last few
        calls before it is refused instead of run; ``max_repeat_refusals`` is how many such
        refusals **in a row** make the runner give up on that call for the rest of the turn
        (:attr:`repeat_wedged`)."""
        self._tools = dict(tools)
        self._context = context
        self._events = events
        self._session_id = session_id
        # Held by reference so hooks appended after the session started take effect.
        self._hooks = hooks if hooks is not None else Hooks()
        self.max_tool_calls = max_tool_calls
        self.max_repeats = max_repeats
        self.max_repeat_refusals = max_repeat_refusals
        self.context_window_tokens = context_window_tokens
        self._lock = threading.Lock()
        self.records: list[ToolCallRecord] = []
        self._turn_records: list[ToolCallRecord] = []
        self._turn_calls = 0
        self._recent: deque[tuple[str, str]] = deque(maxlen=8)
        self.limit_hit = False
        self._refusals_in_a_row = 0
        self._given_up: set[tuple[str, str]] = set()
        self._turn_cancelled = False

    def begin_turn(self) -> None:
        with self._lock:
            self._turn_records = []
            self._turn_calls = 0
            self._recent.clear()
            self.limit_hit = False
            self._refusals_in_a_row = 0
            self._given_up.clear()
            self._turn_cancelled = False

    @property
    def repeat_wedged(self) -> bool:
        """Has the repeated-call guardrail given up on some call this turn?

        Sticky for the turn, and published so a caller can tell a turn that wound down here
        from one that spent its budget (``limit_hit``). Which call it was is in the records.
        """
        return bool(self._given_up)

    @property
    def no_more_tool_rounds(self) -> bool:
        """Should a loop Kennel drives itself stop starting tool rounds this turn?

        Two ways to run out — the per-turn budget (``limit_hit``) and a model wedged on one
        repeated call (``repeat_wedged``) — but one question, because the answer is the same:
        the next request should be the last one. ``ChatLoopSession`` reads this
        (``chatloop._limit_reached``) so the loop keeps no second copy of either rule
        (principle 2).

        It is a question about rounds, not about single calls: a call that changes its
        arguments is still served. That matters for a provider that runs its own tool loop
        (Apple's SDK), which never asks this question and cannot be stopped from a tool
        callback. There, refusing the repeated call while still answering a narrowed one is
        the whole lever.
        """
        return self.limit_hit or self.repeat_wedged

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
            return self._invalid(name, dict(arguments), summary, exc)
        summary = tool.summarize(args)

        with self._lock:
            self._turn_calls += 1
            key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            self._recent.append(key)
            repeats = sum(1 for k in self._recent if k == key)
            over_limit = self._turn_calls > self.max_tool_calls
            if over_limit:
                self.limit_hit = True
            repeated = repeats >= self.max_repeats
            # Counted in a row, and reset by any call that is not a repeat: a model working
            # through a long turn may repeat something twice by accident, but one that has
            # been refused this many times running is not reading the refusal at all.
            self._refusals_in_a_row = self._refusals_in_a_row + 1 if repeated else 0
            if self._refusals_in_a_row >= self.max_repeat_refusals:
                self._given_up.add(key)
            given_up = key in self._given_up
        if over_limit:
            return self._blocked(
                name, args, summary,
                f"tool call limit ({self.max_tool_calls} per turn) reached",
                f"Error: the tool call limit for this turn was reached. {STOP_CALLING_TOOLS}",
            )
        if given_up:
            # Checked before the repeat itself so this call's answer stays terminal for the
            # rest of the turn (issue #58: without this the turn runs until the budget is
            # spent, and the user loses it).
            return self._blocked(
                name, args, summary,
                f"refused {self.max_repeat_refusals} times in a row; not run again this turn",
                TOOLS_STOPPED_NOTICE,
            )
        if repeated:
            return self._blocked(
                name, args, summary, "repeated identical tool call", REPEATED_CALL_REFUSAL
            )

        hook_context: HookContext | None = None
        if self._hooks.before_tool:
            hook_context = HookContext(self._session_id, self._context, tool)
            try:
                outcome = await self._before_tool(tool, args, summary, hook_context)
            except Exception as exc:  # noqa: BLE001 - a broken policy is an error, not silence
                log.exception("before_tool hook failed for %s", name)
                return self._fail(name, args, summary, exc, prefix="hook failed: ")
            if not outcome.allowed:
                self._record(ToolCallRecord(name, args, summary, "blocked", error=outcome.message))
                self._emit(EventType.TOOL_BLOCKED, tool=name, summary=summary, error=outcome.message)
                return f"Error: {outcome.message}"
            if outcome.updated_arguments is not None:
                try:
                    args = tool.validate(outcome.updated_arguments)
                except ToolArgumentError as exc:
                    return self._invalid(name, args, summary, exc)
                summary = tool.summarize(args)

        pm = self._context.permission_manager
        decision = pm.decision_for(name, tool.permission, args, tool.match_rule)
        if decision is not Decision.ALLOW:
            try:
                details = tool.permission_details(args, self._context)
                warnings = tool.permission_warnings(args, self._context)
            except KennelError as exc:
                return self._fail(name, args, summary, exc)
            request = PermissionRequest(
                name, tool.permission, summary, details, warnings, dict(args), tool.session_scope(args)
            )
            self._emit(EventType.PERMISSION_REQUESTED, tool=name, summary=summary, warnings=list(warnings))
            decided = pm.decide(request, matcher=tool.match_rule, decision=decision)
            if not decided.allowed:
                reason = decided.message or "The user did not approve this action; do not retry it, explain what you would have done instead."
                self._record(ToolCallRecord(name, args, summary, "denied", error=decided.message or "permission denied"))
                self._emit(EventType.PERMISSION_DENIED, tool=name, summary=summary, error=decided.message)
                return f"Error: permission denied for {name}. {reason}"
            if decided.updated_arguments is not None:
                try:
                    args = tool.validate(decided.updated_arguments)
                except ToolArgumentError as exc:
                    return self._invalid(name, args, summary, exc)
                summary = tool.summarize(args)

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

        if self._hooks.after_tool:
            if hook_context is None:
                hook_context = HookContext(self._session_id, self._context, tool)
            try:
                result = await self._after_tool(tool, args, summary, result, hook_context)
            except Exception as exc:  # noqa: BLE001 - a broken policy is an error, not silence
                log.exception("after_tool hook failed for %s", name)
                return self._fail(name, args, summary, exc, started, prefix="hook failed: ")

        content, truncated = truncate_text(result.content, self._context.limits.max_output_bytes)
        truncated = truncated or result.truncated
        output_bytes = len(content.encode())
        # An estimate, and named as one: the on-device provider counts no tokens, so this is
        # what lets a caller (and the CLI) see a result that cannot fit before the model chokes
        # on it. The comparison is about this one result, not about the live context.
        estimated_tokens = math.ceil(estimate_tokens_from_bytes(output_bytes))
        window = self.context_window_tokens
        # The output limit bounds every result (``tools.max_output_bytes``, whatever the user
        # set it to); this only asks whether what came through it still cannot fit the window
        # at all. Such a result is not worth handing over: the request fails instead of the
        # model reading it. So it is withheld and the model is told how to ask for less, which
        # keeps the turn alive and leaves the limit, and the meaning of ``truncated``, alone.
        window_exceeded = window is not None and estimated_tokens > window
        duration = (time.perf_counter() - started) * 1000
        record = self._record(
            ToolCallRecord(
                name, args, summary, "ok", output_bytes, truncated, duration, None,
                dict(result.metadata), withheld=window_exceeded,
            )
        )
        self._emit(
            EventType.TOOL_COMPLETED,
            tool=name,
            summary=summary,
            output_bytes=record.output_bytes,
            estimated_tokens=estimated_tokens,
            context_window_tokens=window,
            window_exceeded=window_exceeded,
            withheld=record.withheld,
            truncated=truncated,
            duration_ms=duration,
            metadata=dict(result.metadata),
        )
        if record.withheld:
            return WINDOW_EXCEEDED_NOTICE.format(tool=name, tokens=estimated_tokens, window=window)
        return content

    async def _before_tool(
        self, tool: Tool, args: dict[str, Any], summary: str, context: HookContext
    ) -> PermissionOutcome:
        """Run the hooks in order. The first denial wins; argument rewrites accumulate.

        What an ``Allow`` / ``Deny`` means (including remembering a grant) is
        :meth:`PermissionManager.resolve`'s job, so hooks and prompters cannot
        drift apart.
        """
        pm = self._context.permission_manager
        updated: Mapping[str, Any] | None = None
        call = ToolCallRequest(tool.name, args, summary)
        for hook in select_hooks(self._hooks.before_tool, tool.name):
            answer = await run_hook(hook, call, context)
            outcome = pm.resolve(tool.name, answer, scope=tool.session_scope(call.arguments))
            if not outcome.allowed:
                return outcome
            if outcome.updated_arguments is not None:
                updated = outcome.updated_arguments
                call = ToolCallRequest(tool.name, updated, summary)
        return PermissionOutcome(True, updated_arguments=updated)

    async def _after_tool(
        self, tool: Tool, args: dict[str, Any], summary: str, result: ToolResult, context: HookContext
    ) -> ToolResult:
        call = ToolCallRequest(tool.name, args, summary)
        for hook in select_hooks(self._hooks.after_tool, tool.name):
            replaced = await run_hook(hook, call, result, context)
            if replaced is not None:
                result = replaced
        return result

    def _blocked(self, name: str, args: dict[str, Any], summary: str, reason: str, answer: str) -> str:
        """Record a call a guardrail did not let run, and give the model ``answer`` instead."""
        self._record(ToolCallRecord(name, args, summary, "blocked", error=reason))
        self._emit(EventType.TOOL_FAILED, tool=name, summary=summary, error=reason)
        return answer

    def _invalid(self, name: str, args: dict[str, Any], summary: str, exc: ToolArgumentError) -> str:
        """Record and report arguments that did not validate (raw, or rewritten by a hook)."""
        self._record(ToolCallRecord(name, args, summary, "invalid", error=str(exc)))
        self._emit(EventType.TOOL_FAILED, tool=name, summary=summary, error=str(exc))
        return f"Error: {exc}"

    def _fail(self, name: str, args: dict[str, Any], summary: str, exc: BaseException, started: float | None = None, prefix: str = "") -> str:
        duration = (time.perf_counter() - started) * 1000 if started else 0.0
        message = f"{prefix}{exc}"
        self._record(ToolCallRecord(name, args, summary, "error", duration_ms=duration, error=message))
        self._emit(EventType.TOOL_FAILED, tool=name, summary=summary, error=message, duration_ms=duration)
        return f"Error: {message}"
