"""Deterministic provider for tests and offline demos.

A ``MockProvider`` is scripted per turn. Each turn is a plain string, a list of
steps (:class:`ToolCall`, :class:`Text`, :class:`Raise`, :class:`Sleep`), or an
async callable ``(prompt, invoke) -> str``.

:meth:`MockProvider.from_json` reads the same script from JSON (``KENNEL_MOCK_SCRIPT``):
``{"turns": [...], "structured": [...], "available": false}``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import ModelUnavailableError, ProviderError
from ..tools.base import Tool
from .base import ModelProvider, ProviderInfo, ProviderSession, ToolInvoker, Usage


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Text:
    text: str


@dataclass
class Raise:
    error: BaseException


@dataclass
class Sleep:
    seconds: float


Step = ToolCall | Text | Raise | Sleep
Turn = str | Sequence[Step] | Callable[[str, ToolInvoker], Awaitable[str]]


class MockSession(ProviderSession):
    def __init__(self, provider: MockProvider, instructions: str, tools: Sequence[Tool], invoke: ToolInvoker) -> None:
        self.provider = provider
        self.instructions = instructions
        self.tools = list(tools)
        self.invoke = invoke
        self.prompts: list[str] = []
        self.tool_results: list[str] = []
        self.closed = False

    async def respond(self, prompt: str) -> str:
        self.prompts.append(prompt)
        turn = self.provider._next_turn()
        if turn is None:
            return f"mock response to: {prompt}"
        if isinstance(turn, str):
            return turn
        if callable(turn):
            return await turn(prompt, self.invoke)
        for step in turn:
            if isinstance(step, ToolCall):
                self.tool_results.append(await self.invoke(step.name, dict(step.arguments)))
            elif isinstance(step, Text):
                return step.text
            elif isinstance(step, Raise):
                raise step.error
            elif isinstance(step, Sleep):
                await asyncio.sleep(step.seconds)
        return ""

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        text = await self.respond(prompt)
        for i in range(0, len(text), self.provider.chunk_size):
            yield text[i : i + self.provider.chunk_size]
            await asyncio.sleep(0)

    async def respond_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        if not self.provider._structured:
            raise ProviderError("MockProvider has no structured responses left")
        return self.provider._structured.pop(0)

    def usage(self) -> Usage | None:
        return self.provider.reported_usage

    async def close(self) -> None:
        self.closed = True


class MockProvider(ModelProvider):
    def __init__(
        self,
        turns: Sequence[Turn] | None = None,
        *,
        structured: Sequence[dict[str, Any]] | None = None,
        available: bool = True,
        chunk_size: int = 8,
        context_window_tokens: int | None = 4096,
        reported_usage: Usage | None = None,
    ) -> None:
        """``context_window_tokens=None`` mimics a provider that declares no window;
        ``reported_usage`` mimics one that counts tokens itself."""
        self._turns: list[Turn] = list(turns or [])
        self._structured: list[dict[str, Any]] = list(structured or [])
        self.available = available
        self.chunk_size = chunk_size
        self.reported_usage = reported_usage
        self.info = ProviderInfo(
            name="mock", model="MockModel", mode="local", context_window_tokens=context_window_tokens
        )
        self.sessions: list[MockSession] = []

    def _next_turn(self) -> Turn | None:
        return self._turns.pop(0) if self._turns else None

    def check_availability(self) -> None:
        if not self.available:
            raise ModelUnavailableError("Mock model is marked unavailable")

    async def create_session(self, *, instructions: str, tools: Sequence[Tool], invoke: ToolInvoker) -> ProviderSession:
        session = MockSession(self, instructions, tools, invoke)
        self.sessions.append(session)
        return session

    # -- scripting from JSON (used by `kennel --provider mock`) ---------------

    @classmethod
    def from_json(cls, data: str | dict[str, Any]) -> MockProvider:
        obj = json.loads(data) if isinstance(data, str) else data
        turns: list[Turn] = []
        for turn in obj.get("turns", []):
            if isinstance(turn, str):
                turns.append(turn)
                continue
            steps: list[Step] = []
            for step in turn:
                if "tool" in step:
                    steps.append(ToolCall(step["tool"], dict(step.get("arguments", {}))))
                elif "text" in step:
                    steps.append(Text(step["text"]))
                elif "sleep" in step:
                    steps.append(Sleep(float(step["sleep"])))
            turns.append(steps)
        return cls(turns, structured=obj.get("structured"), available=bool(obj.get("available", True)))
