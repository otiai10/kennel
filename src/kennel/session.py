"""Session: a conversation with the model plus its tool trace."""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from .context import (
    HistoryTurn,
    compact_history,
    estimate_tokens,
    estimate_tokens_from_bytes,
    looks_like_tool_narration,
)
from .errors import ContextLimitError, KennelError, ProviderError, TurnCancelledError
from .events import Event, EventType, JsonlEventLog, session_log_path
from .hooks import run_hook
from .providers.base import ProviderSession, Usage
from .runner import ToolCallRecord, ToolRunner

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class ContextUsage:
    """How much of the model's context window the live conversation occupies.

    ``used_tokens`` counts what is actually in the current provider session:
    the instructions (including a compaction summary, when there is one) plus the
    turns since that session was opened. Compacting therefore lowers it, which is
    the point of showing it.
    """

    window_tokens: int | None
    used_tokens: int
    ratio: float  # 0.0-1.0; 0.0 when the provider declares no window
    estimated: bool
    turns: int
    compactions: int

    def summary(self) -> str:
        """One line for the CLI: ``12% of 4096 tokens (504 tokens, estimated)``."""
        detail = f"{self.used_tokens} tokens" + (", estimated" if self.estimated else "")
        if self.window_tokens is None:
            return f"{detail}, window unknown"
        return f"{self.ratio:.0%} of {self.window_tokens} tokens ({detail})"


@dataclass
class AgentResult:
    """The outcome of one turn. :meth:`to_dict` is the machine-readable form.

    ``is_error`` marks a turn that did not produce a usable answer (a timeout, or
    an error the caller turned into a result). ``tool_limit`` is not an error: the
    answer is there, only possibly incomplete.
    """

    text: str
    stop_reason: str  # end_turn | tool_limit | timeout | error
    tool_calls: list[ToolCallRecord]
    usage: Usage | None
    session_id: str
    duration_ms: float = 0.0
    is_error: bool = False
    structured_output: dict[str, Any] | None = None
    compactions: int = 0
    error: str | None = None
    failure: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable result record (schema: ``docs/output-format.md``)."""
        return {
            "type": "result",
            "text": self.text,
            "stop_reason": self.stop_reason,
            "is_error": self.is_error,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 3),
            "session_id": self.session_id,
            "tool_calls": [asdict(call) for call in self.tool_calls],
            "usage": asdict(self.usage) if self.usage is not None else None,
            "structured_output": self.structured_output,
            "compactions": self.compactions,
            "failure": self.failure,
        }


@dataclass
class Turn:
    """One exchange as it was recorded. ``stop_reason`` ``"error"`` marks a turn that failed."""

    prompt: str
    response: str
    stop_reason: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


NUDGE_PROMPT = (
    "Do not describe, propose or ask about the steps; carry them out now by calling the tools "
    "(start with glob), then give the answer in the language of my previous message. "
    "Do not ask for confirmation or which file to check."
)


class Session:
    def __init__(self, agent: Agent, *, session_id: str | None = None) -> None:
        self.agent = agent
        self.id = session_id or uuid.uuid4().hex[:12]
        self.history: list[Turn] = []
        self.compactions = 0
        self.last_result: AgentResult | None = None
        #: Facts about the turn that failed most recently, as ``session.failed`` carried
        #: them (``None`` until a turn fails). Counted or reported only: see :meth:`_fail_turn`.
        self.last_failure: dict[str, Any] | None = None
        #: Where this session's event log goes, or ``None`` when logging is off
        #: (``logging.events: false`` / ``--no-log``). The file appears with the first turn.
        self.log_path: Path | None = session_log_path(self.id) if agent.config.log_events else None
        self._event_log: JsonlEventLog | None = None
        self._stop_logging: Callable[[], None] | None = None
        self._provider_session: ProviderSession | None = None
        self._reset_window(agent.instructions)
        self._started = False
        self._failed = False
        self._task: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._interrupted = False
        self.runner = ToolRunner(
            agent.tools,
            agent.tool_context(),
            agent.events,
            self.id,
            max_tool_calls=agent.config.max_tool_calls,
            hooks=agent.hooks,
        )

    # -- lifecycle ------------------------------------------------------------

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    def _emit(self, type: str, **data: Any) -> None:
        if type == EventType.SESSION_FAILED:
            self._failed = True
        self.agent.events.emit(type, self.id, **data)

    async def _provider(self, extra_instructions: str = "") -> ProviderSession:
        if self._provider_session is None:
            instructions = self._compose_instructions(extra_instructions)
            self._provider_session = await self.agent.provider.create_session(
                instructions=instructions,
                tools=list(self.agent.tools.values()),
                invoke=self.runner.invoke,
            )
            self._reset_window(instructions)
        return self._provider_session

    def _compose_instructions(self, extra_instructions: str) -> str:
        if not extra_instructions:
            return self.agent.instructions
        return f"{self.agent.instructions}\n\n{extra_instructions}"

    def _reset_window(self, instructions: str) -> None:
        """Point the estimated-window bookkeeping at what the live provider session holds.

        ``instructions`` is the full text that session was opened with — already
        composed by :meth:`_compose_instructions`, not re-derived here, so
        :meth:`_provider` does not pay for it twice.

        Only :meth:`_provider` (and ``__init__``, for the fields to exist) calls this:
        the pair is read by :meth:`_estimate_window_tokens` *while a session is live*,
        and what a dropped session leaves behind is derived there instead of pushed
        here (see :meth:`_drop_provider`).
        """
        self._window_instructions = instructions
        self._window_turn_start = len(self.history)

    async def _drop_provider(self) -> None:
        """Retire the live provider session, if any.

        Every caller — :meth:`close`, :meth:`clear`, :meth:`compact`,
        :meth:`_abandon_turn` and ``run()``'s timeout handler — gets the same
        estimated-window bookkeeping without doing anything extra, because
        :meth:`_estimate_window_tokens` derives the window from whether a session is
        live rather than each caller resetting it (#52, and #45 for the reader that
        must not see the pre-drop size: the failed-turn report, `/status` right after
        Ctrl-C). There is deliberately no exception: a path that wanted to keep
        counting the retired session's history would be counting a transcript the
        provider has already forgotten.
        """
        session, self._provider_session = self._provider_session, None
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001
                pass

    async def close(self) -> None:
        await self._drop_provider()
        if self._started and not self._failed:
            self._emit(EventType.SESSION_COMPLETED, turns=len(self.history))
        self._close_event_log()  # after the closing event, so the log ends with it

    def _open_event_log(self) -> None:
        """Start writing this session's events to :attr:`log_path` (a no-op when logging is off).

        Opened with the first turn rather than with the session, so a session that is never
        run leaves no file behind. Idempotent, and a log that cannot be opened disables
        itself and warns rather than raising.
        """
        if self.log_path is None or self._event_log is not None:
            return
        self._event_log = JsonlEventLog(self.log_path, session_id=self.id)
        self._stop_logging = self.agent.events.subscribe(self._event_log)

    def _close_event_log(self) -> None:
        if self._stop_logging is not None:
            self._stop_logging()
            self._stop_logging = None
        if self._event_log is not None:
            self._event_log.close()
            self._event_log = None

    async def clear(self) -> None:
        """Forget the conversation (keeps configuration and the session id)."""
        self.history.clear()  # before dropping, so the window derives from an empty history
        self.compactions = 0
        await self._drop_provider()

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "turns": len(self.history),
            "tool_calls": len(self.runner.records),
            "compactions": self.compactions,
            "provider": self.agent.provider.info.name,
            "model": self.agent.provider.info.model,
            "mode": self.agent.provider.info.mode,
            "context": self.context_usage().summary(),
            "log": str(self.log_path) if self.log_path is not None else "off",
            "workspace": str(self.agent.workspace.root),
        }

    def interrupt(self) -> None:
        """Stop the turn that is running, keeping the session usable.

        Safe to call from any thread or task: the cancellation is posted to the
        loop that started the turn. The waiting ``run()`` emits
        ``session.cancelled`` and raises
        :class:`~kennel.errors.TurnCancelledError`; a following ``run()`` starts a
        fresh turn on a new provider session. A no-op when no turn is running,
        so a UI can wire it straight to a stop button.
        """
        task, loop = self._task, self._loop
        if task is None or loop is None or task.done():
            return
        self._interrupted = True
        self.runner.cancel_turn()
        self.agent.permissions.cancel_prompt()  # a prompter blocked on stdin would hold the turn
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:  # pragma: no cover - the loop is already closed
            pass

    async def _abandon_turn(self, text: str) -> NoReturn:
        """Wind down a cancelled turn, then raise what the caller should see.

        Call this from an ``except asyncio.CancelledError`` handler; it always
        raises. :meth:`interrupt` asked for the stop, so the turn becomes a
        ``TurnCancelledError``; any other cancellation keeps asyncio's meaning and
        the handled ``CancelledError`` is re-raised as is.
        """
        self.runner.cancel_turn()
        self._emit(EventType.MODEL_COMPLETED, stop_reason="cancelled", chars=len(text))
        await self._drop_provider()
        if self._interrupted:
            self._emit(EventType.SESSION_CANCELLED, turns=len(self.history))
            raise TurnCancelledError("The turn was interrupted.") from None
        raise

    def context_usage(self) -> ContextUsage:
        """Report how full the model's context window is right now.

        Uses the provider's own count when it has one
        (:meth:`~kennel.ProviderSession.usage`) and falls back to estimating from
        the text in the live provider session, in which case ``estimated`` is True.

        Right after an interrupt, a timeout, a failed turn or :meth:`compact`, the
        live provider session has just been retired, so there is nothing left to ask
        :meth:`~kennel.ProviderSession.usage` — ``estimated`` flips to True and
        ``used_tokens`` drops to what the *next* session will open with (a summary of
        the history so far), even on a provider that otherwise reports real counts.
        That is not a lost conversation, just a switch from "what the retired session
        held" to "what the next one will start with".
        """
        window = self.agent.provider.info.context_window_tokens
        measured = self._reported_usage()
        if measured is not None:
            used, estimated = (measured.input_tokens or 0) + (measured.output_tokens or 0), False
        else:
            used, estimated = math.ceil(self._estimate_window_tokens()), True
        ratio = min(1.0, used / window) if window else 0.0
        return ContextUsage(window, used, ratio, estimated, len(self.history), self.compactions)

    def _reported_usage(self) -> Usage | None:
        """What the live provider session counted, when the provider counts at all."""
        return self._provider_session.usage() if self._provider_session is not None else None

    def _estimate_window_tokens(self) -> float:
        """Estimate the tokens held by the live provider session (see :class:`ContextUsage`).

        With no live session, the estimate is what the *next* one will be opened with
        instead: :meth:`_provider` is only ever called with ``_compaction_note()``, so a
        summary of the whole history replaces it verbatim and nothing is carried over.
        Deriving it here rather than resetting the bookkeeping in every
        :meth:`_drop_provider` caller keeps the answer in one place (principle 2) and
        keeps it right for ``run()``'s timeout handler, which drops the session *before*
        appending the timed-out turn to :attr:`history` (#52).
        """
        if self._provider_session is None:
            return estimate_tokens(self._compose_instructions(self._compaction_note()))
        total = estimate_tokens(self._window_instructions)
        for turn in self.history[self._window_turn_start :]:
            total += estimate_tokens(turn.prompt) + estimate_tokens(turn.response)
            # A withheld result never reached the provider session (ToolRunner.invoke gave the
            # model a one-line notice instead), so its bytes are not in the window.
            total += sum(estimate_tokens_from_bytes(c.output_bytes) for c in turn.tool_calls if not c.withheld)
        return total

    # -- failing --------------------------------------------------------------

    async def _fail_turn(
        self,
        exc: BaseException,
        prompt: str,
        text: str,
        started: float,
        before: ContextUsage,
    ) -> None:
        """Record a turn that failed, retire the provider session and emit ``session.failed``.

        The turn lands in :attr:`history` with ``stop_reason="error"`` and the tool calls
        it did make, so ``/usage``, ``/status`` and :meth:`context_usage` stop pretending
        it never happened. The live provider session is retired through :meth:`compact`
        because the provider gives no way to tell whether the failed prompt and its tool
        output stayed in its transcript; the next turn resumes from a summary instead of
        from an undefined state.

        Callers must raise afterwards: this reports the failure, it does not swallow it.
        """
        records = self.runner.turn_records()
        self.history.append(Turn(prompt, text, "error", records))
        failure = {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "status": getattr(exc, "status", None),
            "provider_error": getattr(exc, "provider_error", None),
            "tool_output_bytes": sum(record.output_bytes for record in records),
            "tool_calls": len(records),
            "context_tokens_before": before.used_tokens,
            "context_window_tokens": before.window_tokens,
            "estimated": before.estimated,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        self.last_failure = failure
        if self._provider_session is not None:
            await self.compact()
        self._emit(EventType.SESSION_FAILED, **failure)

    # -- running --------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        *,
        on_delta: Callable[[str], None] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Send ``prompt`` and return the final answer once the model stops calling tools.

        With ``on_delta`` the answer is streamed and the callback receives text
        deltas as they arrive (also emitted as ``model.delta`` events).

        :meth:`interrupt` stops the turn in flight, in which case this raises
        :class:`~kennel.errors.TurnCancelledError`.

        With ``schema`` (a JSON schema) the model answers under guided generation:
        the value lands in :attr:`AgentResult.structured_output` and its JSON text in
        :attr:`AgentResult.text`. Tools still work — the provider calls them inside the
        same request — so ``tool_calls`` is populated as usual.
        """
        if not prompt.strip():
            raise KennelError("Prompt must not be empty")
        self._task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        self._interrupted = False
        self._open_event_log()  # idempotent; before the first event so it lands in the file
        if not self._started:
            self._started = True
            self._emit(
                EventType.SESSION_STARTED,
                workspace=str(self.agent.workspace.root),
                model=self.agent.provider.info.model,
                tools=sorted(self.agent.tools),
            )
        self.runner.begin_turn()
        # Read per turn rather than captured once: a provider may only learn its window in
        # check_availability() (llama-server asks the server), and context_usage() reads it
        # the same way, so the two can never disagree.
        self.runner.context_window_tokens = self.agent.provider.info.context_window_tokens
        self.last_failure = None
        self._emit(EventType.MODEL_STARTED, prompt_chars=len(prompt))
        started = time.perf_counter()
        before = self.context_usage()  # what the window held before this request, for _fail_turn
        stop_reason = "end_turn"
        text = ""
        attempt = 0
        timeout = self.agent.config.turn_timeout_seconds
        # before_prompt applies to what the user asked, not to Kennel's own
        # re-prompts, and the history keeps the user's words.
        sent = await self._apply_before_prompt(prompt)
        structured: dict[str, Any] | None = None

        async def generate() -> str:
            nonlocal structured
            if schema is None:
                return await self._generate(sent, on_delta)
            provider = await self._provider(self._compaction_note())
            data = await provider.respond_structured(sent, schema)
            structured = data
            # Guided generation arrives whole, so the answer is one delta.
            answer = json.dumps(data, ensure_ascii=False, indent=2)
            self._emit(EventType.MODEL_DELTA, chars=len(answer))
            if on_delta is not None:
                on_delta(answer)
            return answer

        while True:
            try:
                text = await asyncio.wait_for(generate(), timeout)
                break
            except ContextLimitError as exc:
                if attempt >= 1 or not self.history:
                    failed = ContextLimitError(
                        "The request does not fit the on-device model's context window, even after compacting "
                        "the conversation. Ask about a narrower range or fewer files.",
                        status=exc.status,
                        provider_error=exc.provider_error,
                    )
                    await self._fail_turn(failed, prompt, text, started, before)
                    raise failed from None
                attempt += 1
                await self.compact()
            except asyncio.TimeoutError:
                stop_reason = "timeout"
                await self._drop_provider()
                break
            except asyncio.CancelledError:
                await self._abandon_turn(text)
            except KennelError as exc:
                await self._fail_turn(exc, prompt, text, started, before)
                raise
            except BrokenPipeError:
                raise  # the consumer's output went away; not a model failure
            except Exception as exc:  # noqa: BLE001 - provider bug surfaced as a KennelError
                failed = ProviderError(f"The model request failed: {exc}", provider_error=str(exc))
                await self._fail_turn(failed, prompt, text, started, before)
                raise failed from exc
        nudge_rule = self._should_nudge(text) if schema is None and stop_reason == "end_turn" else None
        if nudge_rule is not None:
            # The model narrated tool steps instead of taking them: continue once.
            self._emit(EventType.MODEL_NUDGED, chars=len(text), rule=nudge_rule)
            try:
                text = await asyncio.wait_for(self._generate(NUDGE_PROMPT, on_delta), timeout)
            except asyncio.CancelledError:
                await self._abandon_turn(text)
            except (ContextLimitError, asyncio.TimeoutError):
                pass  # keep the narrated answer rather than fail the turn
        if stop_reason == "end_turn" and self.runner.limit_hit:
            stop_reason = "tool_limit"
        records = self.runner.turn_records()
        self.history.append(Turn(prompt, text, stop_reason, records))
        self._emit(EventType.MODEL_COMPLETED, stop_reason=stop_reason, chars=len(text), tool_calls=len(records))
        # usage stays None unless the provider counts tokens: an estimate per turn would
        # duplicate context_usage() without anything to calibrate it against.
        self.last_result = AgentResult(
            text=text,
            stop_reason=stop_reason,
            tool_calls=records,
            usage=self._reported_usage(),
            session_id=self.id,
            duration_ms=(time.perf_counter() - started) * 1000,
            is_error=stop_reason == "timeout",
            structured_output=structured,
            compactions=self.compactions,
        )
        return self.last_result

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Run ``prompt`` and yield this session's events on the calling event loop.

        The turn runs as a task while events are relayed through an
        :class:`asyncio.Queue`, so tool callbacks firing on a provider worker
        thread still reach the consumer's loop. The iterator ends with a
        ``session.completed`` event carrying the final ``text``, ``stop_reason``
        and ``tool_calls`` count; :attr:`last_result` holds the full
        :class:`AgentResult`. A provider failure is raised out of the iterator
        after the ``session.failed`` event has been yielded.

        Example::

            async with agent.new_session() as session:
                async for event in session.stream("Summarize today's transcript"):
                    print(event.type, event.data)
                print(session.last_result.text)
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        def put(event: Event | None) -> None:
            try:
                if asyncio.get_running_loop() is loop:
                    queue.put_nowait(event)
                    return
            except RuntimeError:
                pass  # emitted from a provider worker thread: hand it to the consumer's loop
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:  # pragma: no cover - the consumer's loop went away
                pass

        def relay(event: Event) -> None:
            # Deltas are re-emitted below with their text; the bus only carries sizes.
            if event.session_id == self.id and event.type != EventType.MODEL_DELTA:
                put(event)

        def on_delta(delta: str) -> None:
            put(Event(EventType.MODEL_DELTA, self.id, {"chars": len(delta), "text": delta}))

        unsubscribe = self.agent.events.subscribe(relay)
        task = loop.create_task(self.run(prompt, on_delta=on_delta))
        task.add_done_callback(lambda _: put(None))
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            result = await task  # re-raises after session.failed has been yielded
            yield Event(
                EventType.SESSION_COMPLETED,
                self.id,
                {
                    "turns": len(self.history),
                    "stop_reason": result.stop_reason,
                    "tool_calls": len(result.tool_calls),
                    "text": result.text,
                },
            )
        finally:
            unsubscribe()
            if not task.done():
                task.cancel()


    async def _apply_before_prompt(self, prompt: str) -> str:
        """Append whatever the ``before_prompt`` hooks add to the prompt (additional context)."""
        if not self.agent.hooks.before_prompt:
            return prompt
        extra: list[str] = []
        for hook in list(self.agent.hooks.before_prompt):
            added = await run_hook(hook, prompt, self)
            if added:
                extra.append(str(added).strip())
        return "\n\n".join([prompt, *extra]) if extra else prompt

    def _should_nudge(self, text: str) -> str | None:
        """The narration rule that fired, or ``None`` if this turn should not be nudged.

        A turn that already called a tool (this one, or immediately before it) is
        answering from what it just saw, not narrating unseen steps: nudging it
        would re-run the same tools and double the answer for no reason (#33).

        This also suppresses the nudge right after a turn that called tools and then
        *failed* (``stop_reason == "error"``, recorded by :meth:`_fail_turn`): the check
        below only looks at whether that turn made tool calls, not whether it completed.
        This is deliberate, not an oversight (#46): re-running the tools that a failed
        turn already called is exactly what is likely to overflow the context window
        again, so staying quiet is the safe default. The alternative — nudging because
        the model "hasn't really answered from a result yet" — was considered and
        rejected precisely because it would retry into the same failure.
        """
        if not self.agent.tools or not self.agent.config.nudge_narration:
            return None
        if self.runner.turn_records():
            return None
        if self.history and self.history[-1].tool_calls:
            return None
        return looks_like_tool_narration(text, self.agent.tools)

    async def _generate(self, prompt: str, on_delta: Callable[[str], None] | None) -> str:
        provider = await self._provider(self._compaction_note())
        if on_delta is None:
            return await provider.respond(prompt)
        parts: list[str] = []
        async for delta in provider.stream(prompt):
            parts.append(delta)
            self._emit(EventType.MODEL_DELTA, chars=len(delta))
            on_delta(delta)
        return "".join(parts)

    def _compaction_note(self) -> str:
        if self._provider_session is not None or not self.history:
            return ""
        return compact_history([HistoryTurn(t.prompt, t.response) for t in self.history])

    async def compact(self) -> bool:
        """Replace the provider session with a fresh one seeded by a compact summary of the history.

        Returns ``False`` (no-op, no event emitted) if there is no history to compact.
        Retiring the provider session is enough to move the estimated-window bookkeeping
        (:meth:`context_usage`) to the size of the summary rather than the history it
        replaces (#45, #52) — a caller that reads it right after (a failed turn's report,
        `/status`) before the next `_provider()` call sees the post-compaction size, not
        the pre-compaction one.
        """
        if not self.history:
            return False
        await self._drop_provider()
        self.compactions += 1
        self._emit(EventType.CONTEXT_COMPACTED, turns=len(self.history))
        return True
