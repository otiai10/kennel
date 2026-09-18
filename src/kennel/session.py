"""Session: a conversation with the model plus its tool trace."""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

from .context import (
    HistoryTurn,
    compact_history,
    estimate_tokens,
    estimate_tokens_from_bytes,
    looks_like_tool_narration,
)
from .errors import ContextLimitError, KennelError, ProviderError, TurnCancelledError
from .events import Event, EventType
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
    stop_reason: str  # end_turn | tool_limit | timeout
    tool_calls: list[ToolCallRecord]
    usage: Usage | None
    session_id: str
    duration_ms: float = 0.0
    is_error: bool = False
    structured_output: dict[str, Any] | None = None
    compactions: int = 0
    error: str | None = None

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
        }


@dataclass
class Turn:
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
        self._provider_session: ProviderSession | None = None
        self._window_instructions = agent.instructions
        self._window_turn_start = 0
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
            instructions = self.agent.instructions
            if extra_instructions:
                instructions = f"{instructions}\n\n{extra_instructions}"
            self._provider_session = await self.agent.provider.create_session(
                instructions=instructions,
                tools=list(self.agent.tools.values()),
                invoke=self.runner.invoke,
            )
            # What the live window holds, for context_usage(): these instructions
            # plus whatever turns follow. Earlier turns are only in the summary.
            self._window_instructions = instructions
            self._window_turn_start = len(self.history)
        return self._provider_session

    async def _drop_provider(self) -> None:
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

    async def clear(self) -> None:
        """Forget the conversation (keeps configuration and the session id)."""
        await self._drop_provider()
        self.history.clear()
        self.compactions = 0
        self._window_instructions = self.agent.instructions
        self._window_turn_start = 0

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "turns": len(self.history),
            "tool_calls": len(self.runner.records),
            "compactions": self.compactions,
            "model": self.agent.provider.info.model,
            "mode": self.agent.provider.info.mode,
            "context": self.context_usage().summary(),
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
        """Estimate the tokens held by the live provider session (see :class:`ContextUsage`)."""
        total = estimate_tokens(self._window_instructions)
        for turn in self.history[self._window_turn_start :]:
            total += estimate_tokens(turn.prompt) + estimate_tokens(turn.response)
            total += sum(estimate_tokens_from_bytes(c.output_bytes) for c in turn.tool_calls)
        return total

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
        if not self._started:
            self._started = True
            self._emit(
                EventType.SESSION_STARTED,
                workspace=str(self.agent.workspace.root),
                model=self.agent.provider.info.model,
                tools=sorted(self.agent.tools),
            )
        self.runner.begin_turn()
        self._emit(EventType.MODEL_STARTED, prompt_chars=len(prompt))
        started = time.perf_counter()
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
            except ContextLimitError:
                if attempt >= 1 or not self.history:
                    self._emit(EventType.SESSION_FAILED, error="context limit")
                    raise ContextLimitError(
                        "The request does not fit the on-device model's context window, even after compacting "
                        "the conversation. Ask about a narrower range or fewer files."
                    ) from None
                attempt += 1
                await self.compact()
            except asyncio.TimeoutError:
                stop_reason = "timeout"
                await self._drop_provider()
                break
            except asyncio.CancelledError:
                await self._abandon_turn(text)
            except KennelError as exc:
                self._emit(EventType.SESSION_FAILED, error=str(exc))
                raise
            except BrokenPipeError:
                raise  # the consumer's output went away; not a model failure
            except Exception as exc:  # noqa: BLE001 - provider bug surfaced as a KennelError
                self._emit(EventType.SESSION_FAILED, error=str(exc))
                raise ProviderError(f"The model request failed: {exc}") from exc
        if schema is None and stop_reason == "end_turn" and self._should_nudge(text):
            # The model narrated tool steps instead of taking them: continue once.
            self._emit(EventType.MODEL_NUDGED, chars=len(text))
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

    def _should_nudge(self, text: str) -> bool:
        if not self.agent.tools or not self.agent.config.nudge_narration:
            return False
        if self.runner.turn_records():
            return False
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
        """
        if not self.history:
            return False
        await self._drop_provider()
        self.compactions += 1
        self._emit(EventType.CONTEXT_COMPACTED, turns=len(self.history))
        return True
