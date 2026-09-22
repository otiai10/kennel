"""The one tool-calling loop Kennel drives itself, for providers that do not.

Apple's SDK runs the loop natively (``providers/apple.py``); a chat-completions
model does not, so something has to parse the model's tool calls, invoke them and
feed the results back. That loop lives here and nowhere else: a provider supplies
a :class:`ChatTransport` (messages in, :class:`ChatChunk` s out) and
:class:`ChatLoopSession` does the rest, so every chat-style provider shares one
implementation of the termination rules, the message bookkeeping and the
streaming contract.

Design notes:

* **Always streaming.** ``ChatTransport.stream`` is the only verb, so there is no
  second, non-streaming code path to keep in sync. ``respond``, ``stream`` and
  ``respond_structured`` are three thin wrappers around one loop, which is what
  makes ``respond()`` and ``"".join(stream())`` return the same text -- including
  any narration the model emits before a tool call.
* **The loop always stops.** When tool calls run out lives in ``ToolRunner``
  (principle 2) -- the per-turn budget, or a model wedged repeating one call -- so the
  loop reads that one question rather than keeping a second copy of either rule; see
  :func:`_limit_reached`. Once they have run out, one more request is allowed (the model
  usually answers from what it has) and a further tool request ends the turn.
* **Nothing reaches a tool except through** ``invoke``, so malformed tool
  arguments are answered with an error message instead of a guessed ``{}``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import ProviderError
from ..tools.base import Tool
from .base import ProviderSession, ToolInvoker, Usage


@dataclass
class ToolCall:
    """A tool call the model asked for, with its arguments still as JSON text."""

    id: str
    name: str
    arguments: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class ToolCallDelta:
    """A fragment of a tool call: ``index`` identifies which call it belongs to."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class ChatMessage:
    """One entry of the conversation, in the shape chat-completions APIs expect."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The wire form, without the keys this message does not use."""
        out: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass
class ChatChunk:
    """One step of a streamed completion.

    ``text`` is answer text only: a thinking model's scratchpad (llama.cpp's
    ``reasoning_content``) is dropped by the transport and never arrives here.
    ``finish_reason`` is what the provider reported, for transports and tests; the
    loop keys off the tool calls it actually accumulated, so a stream claiming
    ``tool_calls`` without any cannot make it spin.
    """

    text: str = ""
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None


class ChatTransport(Protocol):
    """One chat-completions request. Implementations do no tool handling at all."""

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        schema: dict[str, Any] | None,
    ) -> AsyncIterator[ChatChunk]:
        """Send ``messages`` and yield the completion as it arrives.

        ``tools`` are already in wire form (:func:`tool_to_openai`); ``schema``, when
        given, constrains the answer to that JSON schema.
        """
        ...

    async def close(self) -> None: ...


def tool_to_openai(tool: Tool) -> dict[str, Any]:
    """Describe a Kennel tool the way chat-completions APIs expect.

    ``ToolParameter.type`` already carries the JSON-schema type names, so this is a
    pure rename -- unlike Apple's bridge, which needs Python classes.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {"type": param.type, "description": param.description}
                    for param in tool.parameters
                },
                "required": [param.name for param in tool.parameters if param.required],
            },
        },
    }


def _limit_reached(invoke: ToolInvoker) -> bool:
    """Should the next request be the last one -- has the tool runner behind ``invoke``
    run out of tool calls for this turn?

    ``ModelProvider.create_session`` hands a provider a bare callable, so the loop
    asks the object that callable is bound to: with Kennel's own invoker that is
    ``ToolRunner.invoke``, whose ``ToolRunner`` publishes ``no_more_tool_rounds`` -- the
    per-turn budget being spent, or the model being wedged on one repeated call, which the
    loop handles the same way. An invoker that publishes only the older budget flag
    ``limit_hit`` still works, and either may be a callable that computes it. Any other
    invoker -- a plain function, a test double -- has no ``__self__`` carrying either
    attribute and therefore always reads ``False``, meaning "nothing to run out of":
    pass ``limit_hit=`` to :class:`ChatLoopSession` to supply one.
    """
    owner = getattr(invoke, "__self__", None)
    limit = getattr(owner, "no_more_tool_rounds", getattr(owner, "limit_hit", False))
    return bool(limit() if callable(limit) else limit)


def _merge_delta(calls: dict[int, ToolCall], delta: ToolCallDelta) -> None:
    """Fold a streamed fragment into the call at its ``index``.

    Only the first fragment of a call carries ``id`` and ``name``; the rest extend
    the argument text.
    """
    call = calls.get(delta.index)
    if call is None:
        calls[delta.index] = ToolCall(delta.id or "", delta.name or "", delta.arguments)
        return
    if delta.id:
        call.id = delta.id
    if delta.name:
        call.name = delta.name
    call.arguments += delta.arguments


class ChatLoopSession(ProviderSession):
    """A conversation with a chat-completions model, tool loop included.

    ``messages`` is the live conversation (the instructions as a ``system`` message,
    then every turn's prompt, answer, tool call and tool result), which is what a
    fresh provider session after :meth:`kennel.Session.compact` leaves behind.
    """

    def __init__(
        self,
        transport: ChatTransport,
        *,
        instructions: str,
        tools: Sequence[Tool],
        invoke: ToolInvoker,
        limit_hit: Callable[[], bool] | None = None,
    ) -> None:
        self._transport = transport
        self._invoke = invoke
        self._tools = [tool_to_openai(tool) for tool in tools]
        self._limit_hit = limit_hit if limit_hit is not None else lambda: _limit_reached(invoke)
        self.messages: list[ChatMessage] = []
        if instructions:
            self.messages.append(ChatMessage("system", instructions))
        self._usage: Usage | None = None

    async def respond(self, prompt: str) -> str:
        return "".join([delta async for delta in self._iterate(prompt, None)])

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async for delta in self._iterate(prompt, None):
            yield delta

    async def respond_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        text = "".join([delta async for delta in self._iterate(prompt, schema)])
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise ProviderError(f"The model's guided answer was not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ProviderError(f"The model's guided answer was {type(value).__name__}, not an object")
        return value

    def usage(self) -> Usage | None:
        """What the provider counted for the latest request (principle 4: measured, not guessed)."""
        return self._usage

    async def close(self) -> None:
        await self._transport.close()

    async def _iterate(self, prompt: str, schema: dict[str, Any] | None) -> AsyncIterator[str]:
        """Run one turn, yielding answer text as it arrives.

        Requests repeat while the model asks for tools. ``schema`` goes with every
        request, because which request turns out to be the last one is not knowable
        in advance.
        """
        self.messages.append(ChatMessage("user", prompt))
        last_round = False
        while True:
            parts: list[str] = []
            calls: dict[int, ToolCall] = {}
            async for chunk in self._transport.stream(self.messages, self._tools, schema):
                if chunk.usage is not None:
                    self._usage = chunk.usage
                if chunk.text:
                    parts.append(chunk.text)
                    yield chunk.text
                for delta in chunk.tool_calls:
                    _merge_delta(calls, delta)
            text = "".join(parts)
            requested = [calls[index] for index in sorted(calls)]
            if not requested:
                self.messages.append(ChatMessage("assistant", text))
                return
            if last_round:
                # Tool calls have run out and the model is still asking. Stop here, and
                # leave the request out of the conversation: a stored assistant message
                # whose tool calls have no replies is not a conversation a model can be
                # asked to continue. Session.run reports a spent budget as stop_reason
                # "tool_limit"; a turn stopped by the repeated-call guardrail ends as
                # "end_turn", with the guardrail's own records in the trace.
                # What this does not cover: a model that asks for tools in this last round
                # too leaves the turn with no answer text at all (and, under
                # respond_structured, with a ProviderError from parsing ""). That predates
                # issue #58 on the budget path; closing it means deciding what stop_reason
                # says about an answerless turn, which docs/output-format.md fixes as an
                # external contract.
                return
            self.messages.append(ChatMessage("assistant", text or None, tool_calls=requested))
            for call in requested:
                result = await self._tool_result(call)
                self.messages.append(ChatMessage("tool", result, tool_call_id=call.id))
            last_round = self._limit_hit()

    async def _tool_result(self, call: ToolCall) -> str:
        """Invoke one tool call, or tell the model why its request was unusable.

        A call Kennel cannot read is never forwarded: ``ToolRunner`` would have to
        judge arguments the model did not send.
        """
        if not call.name:
            return "Error: that tool call named no tool. Call a tool by name."
        try:
            arguments = json.loads(call.arguments) if call.arguments.strip() else {}
        except ValueError as exc:
            return (
                f"Error: invalid JSON arguments for {call.name}: {exc}. "
                "Send the arguments again as a single JSON object."
            )
        if not isinstance(arguments, dict):
            return (
                f"Error: arguments for {call.name} must be a JSON object, "
                f"got {type(arguments).__name__}."
            )
        return await self._invoke(call.name, arguments)
