"""Provider abstraction: everything model-specific lives behind these classes.

The tool-calling loop is driven by the provider (Apple's SDK runs it natively).
Kennel hands the provider its tools plus an ``invoke`` callback; the provider
calls ``invoke(tool_name, arguments)`` whenever the model requests a tool and
feeds the returned string back to the model. ``invoke`` may be called from any
thread and any event loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import ProviderError
from ..tools.base import Tool

ToolInvoker = Callable[[str, dict[str, Any]], Awaitable[str]]


@dataclass
class Usage:
    """Tokens a request consumed. ``estimated`` marks a Kennel-side approximation."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated: bool = False


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    model: str
    mode: str = "local"  # "local": inference and data stay on this machine
    context_window_tokens: int | None = None  # None: the provider does not declare one


class ProviderSession(ABC):
    """One conversation with the model. Not safe for concurrent requests."""

    @abstractmethod
    async def respond(self, prompt: str) -> str:
        """Return the final text answer (tool calls are handled inside)."""

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield text deltas; the default just yields the whole answer once."""
        yield await self.respond(prompt)

    async def respond_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Guided generation against a JSON schema. Optional."""
        raise ProviderError("This provider does not support structured generation")

    def usage(self) -> Usage | None:
        """Tokens this conversation has consumed so far, if the provider counts them.

        Optional: ``None`` means Kennel has to estimate. Synchronous because a
        provider that tracks this has the number already.
        """
        return None

    async def close(self) -> None:
        return None


class ModelProvider(ABC):
    info: ProviderInfo

    def check_availability(self) -> None:
        """Raise :class:`~kennel.errors.ModelUnavailableError` if the model cannot be used."""
        return None

    @abstractmethod
    async def create_session(
        self, *, instructions: str, tools: Sequence[Tool], invoke: ToolInvoker
    ) -> ProviderSession: ...
