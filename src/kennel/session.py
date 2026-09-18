"""Session: a conversation with the model plus its tool trace."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

from .context import HistoryTurn, compact_history, looks_like_tool_narration
from .errors import ContextLimitError, KennelError, ProviderError, TurnCancelledError
from .events import Event, EventType
from .providers.base import ProviderSession
from .runner import ToolCallRecord, ToolRunner

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class AgentResult:
    text: str
    stop_reason: str  # end_turn | tool_limit | timeout
    tool_calls: list[ToolCallRecord]
    usage: Usage | None
    session_id: str


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

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "turns": len(self.history),
            "tool_calls": len(self.runner.records),
            "compactions": self.compactions,
            "model": self.agent.provider.info.model,
            "mode": self.agent.provider.info.mode,
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

    # -- running --------------------------------------------------------------

    async def run(self, prompt: str, *, on_delta: Callable[[str], None] | None = None) -> AgentResult:
        """Send ``prompt`` and return the final answer once the model stops calling tools.

        With ``on_delta`` the answer is streamed and the callback receives text
        deltas as they arrive (also emitted as ``model.delta`` events).

        :meth:`interrupt` stops the turn in flight, in which case this raises
        :class:`~kennel.errors.TurnCancelledError`.
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
        stop_reason = "end_turn"
        text = ""
        attempt = 0
        timeout = self.agent.config.turn_timeout_seconds
        while True:
            try:
                text = await asyncio.wait_for(self._generate(prompt, on_delta), timeout)
                break
            except ContextLimitError:
                if attempt >= 1 or not self.history:
                    self._emit(EventType.SESSION_FAILED, error="context limit")
                    raise ContextLimitError(
                        "The request does not fit the on-device model's context window, even after compacting "
                        "the conversation. Ask about a narrower range or fewer files."
                    ) from None
                attempt += 1
                await self._compact()
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
        if stop_reason == "end_turn" and self._should_nudge(text):
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
        self.last_result = AgentResult(
            text=text, stop_reason=stop_reason, tool_calls=records, usage=None, session_id=self.id
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

    async def _compact(self) -> None:
        """Replace the provider session with a fresh one seeded by a compact history."""
        await self._drop_provider()
        self.compactions += 1
        self._emit(EventType.CONTEXT_COMPACTED, turns=len(self.history))
