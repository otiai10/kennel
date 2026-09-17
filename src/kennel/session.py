"""Session: a conversation with the model plus its tool trace."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .context import HistoryTurn, compact_history, looks_like_tool_narration
from .errors import ContextLimitError, KennelError, ProviderError
from .events import EventType
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
        self._provider_session: ProviderSession | None = None
        self._started = False
        self._failed = False
        self.runner = ToolRunner(
            agent.tools,
            agent.tool_context(),
            agent.events,
            self.id,
            max_tool_calls=agent.config.max_tool_calls,
        )

    # -- lifecycle ------------------------------------------------------------

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

    # -- running --------------------------------------------------------------

    async def run(self, prompt: str, *, on_delta: Callable[[str], None] | None = None) -> AgentResult:
        """Send ``prompt`` and return the final answer once the model stops calling tools.

        With ``on_delta`` the answer is streamed and the callback receives text
        deltas as they arrive (also emitted as ``model.delta`` events).
        """
        if not prompt.strip():
            raise KennelError("Prompt must not be empty")
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
                await self.compact()
            except asyncio.TimeoutError:
                stop_reason = "timeout"
                await self._drop_provider()
                break
            except asyncio.CancelledError:
                self.runner.cancel_turn()
                self._emit(EventType.MODEL_COMPLETED, stop_reason="cancelled", chars=len(text))
                await self._drop_provider()
                raise
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
                self.runner.cancel_turn()
                self._emit(EventType.MODEL_COMPLETED, stop_reason="cancelled", chars=len(text))
                await self._drop_provider()
                raise
            except (ContextLimitError, asyncio.TimeoutError):
                pass  # keep the narrated answer rather than fail the turn
        if stop_reason == "end_turn" and self.runner.limit_hit:
            stop_reason = "tool_limit"
        records = self.runner.turn_records()
        self.history.append(Turn(prompt, text, stop_reason, records))
        self._emit(EventType.MODEL_COMPLETED, stop_reason=stop_reason, chars=len(text), tool_calls=len(records))
        return AgentResult(text=text, stop_reason=stop_reason, tool_calls=records, usage=None, session_id=self.id)

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
