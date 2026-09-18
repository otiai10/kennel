"""The shared chat tool loop (issue #36), driven by a scripted transport.

No HTTP here: the transport is replaced by a script of chunks, so these tests pin
the loop itself -- how it merges streamed tool-call fragments, what it puts in the
conversation, what it yields, and when it stops. The wire format is pinned in
``test_llama_server.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from kennel import Agent, KennelConfig, ProviderError, Usage
from kennel.providers.base import ModelProvider, ProviderInfo, ProviderSession, ToolInvoker
from kennel.providers.chatloop import (
    ChatChunk,
    ChatLoopSession,
    ChatMessage,
    ToolCallDelta,
    tool_to_openai,
)
from kennel.registry import builtin_registry
from kennel.tools.base import Tool


@dataclass
class Request:
    """What the transport was handed for one request."""

    messages: list[ChatMessage]
    tools: list[dict[str, Any]]
    schema: dict[str, Any] | None


class ScriptedTransport:
    """Replays one list of :class:`ChatChunk` per request and records every request."""

    def __init__(self, script: Sequence[Sequence[ChatChunk]]) -> None:
        self._script = [list(chunks) for chunks in script]
        self.calls: list[Request] = []
        self.closed = False

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        schema: dict[str, Any] | None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(Request([replace(m) for m in messages], list(tools), schema))
        if not self._script:
            raise AssertionError(f"the transport ran out of script after {len(self.calls)} requests")
        for chunk in self._script.pop(0):
            await asyncio.sleep(0)  # a real transport always suspends
            yield chunk

    async def close(self) -> None:
        self.closed = True


class ScriptedProvider(ModelProvider):
    """A provider whose only job is to put a ChatLoopSession on a scripted transport."""

    info = ProviderInfo(name="scripted", model="ScriptedModel", context_window_tokens=4096)

    def __init__(self, transport: ScriptedTransport) -> None:
        self.transport = transport
        self.sessions: list[ChatLoopSession] = []

    async def create_session(
        self, *, instructions: str, tools: Sequence[Tool], invoke: ToolInvoker
    ) -> ProviderSession:
        session = ChatLoopSession(
            self.transport, instructions=instructions, tools=tools, invoke=invoke
        )
        self.sessions.append(session)
        return session


class Recorder:
    """A bare invoker: no ToolRunner behind it, so the loop reads no tool budget."""

    def __init__(self, result: str = "tool output") -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result = result

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return self.result


def tool_round(call_id: str, name: str, arguments: str, *, preamble: str = "") -> list[ChatChunk]:
    """One request that ends in a tool call, split the way a real stream splits it."""
    chunks = [ChatChunk(text=preamble)] if preamble else []
    chunks.append(ChatChunk(tool_calls=[ToolCallDelta(0, call_id, name, arguments[:1])]))
    chunks.append(ChatChunk(tool_calls=[ToolCallDelta(0, arguments=arguments[1:])]))
    chunks.append(ChatChunk(finish_reason="tool_calls"))
    return chunks


def text_round(*pieces: str, usage: Usage | None = None) -> list[ChatChunk]:
    chunks = [ChatChunk(text=piece) for piece in pieces]
    chunks.append(ChatChunk(finish_reason="stop"))
    if usage is not None:
        chunks.append(ChatChunk(usage=usage))  # the include_usage chunk trails the answer
    return chunks


def scripted_agent(meeting_ws: Path, script, **kw) -> tuple[Agent, ScriptedTransport, ScriptedProvider]:
    transport = ScriptedTransport(script)
    provider = ScriptedProvider(transport)
    kw.setdefault("tools", ["glob", "grep", "read"])
    return Agent(meeting_ws, provider=provider, **kw), transport, provider


# -- AC-1: the loop invokes each requested tool and records the exchange ----------------


async def test_two_tool_rounds_then_text(meeting_ws: Path):
    agent, transport, provider = scripted_agent(
        meeting_ws,
        [
            tool_round("call-1", "read", '{"path": "notes/todo.md"}'),
            tool_round("call-2", "grep", '{"pattern": "budget"}'),
            text_round("Final ", "answer."),
        ],
    )
    result = await agent.run("what did we decide?")

    assert result.text == "Final answer."
    assert [call.name for call in result.tool_calls] == ["read", "grep"]
    assert all(call.status == "ok" for call in result.tool_calls)
    assert len(transport.calls) == 3

    messages = provider.sessions[0].messages
    tool_messages = [m for m in messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["call-1", "call-2"]
    assert all(m.content for m in tool_messages)  # the tool output went back to the model
    # The conversation the second request saw: ... user, assistant(tool_calls), tool.
    assert [m.role for m in transport.calls[1].messages] == ["system", "user", "assistant", "tool"]
    assistant = transport.calls[1].messages[2]
    assert [(c.id, c.name) for c in assistant.tool_calls] == [("call-1", "read")]
    # Fragments merged: the arguments arrived one character at a time.
    assert assistant.tool_calls[0].arguments == '{"path": "notes/todo.md"}'


async def test_tools_are_described_in_wire_form(meeting_ws: Path):
    agent, transport, _ = scripted_agent(meeting_ws, [text_round("ok")], tools=["read"])
    await agent.run("hi")
    read = builtin_registry().resolve(["read"], [])[0]
    assert transport.calls[0].tools == [tool_to_openai(read)]
    function = transport.calls[0].tools[0]["function"]
    assert function["name"] == "read" and function["parameters"]["required"] == ["path"]
    assert function["parameters"]["properties"]["start_line"]["type"] == "integer"


async def test_invalid_json_arguments_are_answered_not_invoked():
    recorder = Recorder()
    transport = ScriptedTransport(
        [tool_round("call-1", "read", "{not json"), text_round("recovered")]
    )
    session = ChatLoopSession(
        transport, instructions="sys", tools=[], invoke=recorder.invoke
    )
    assert await session.respond("go") == "recovered"
    assert recorder.calls == []  # nothing reached the tool boundary
    reply = [m for m in session.messages if m.role == "tool"][0]
    assert "invalid JSON arguments for read" in (reply.content or "")


# -- AC-2: streaming yields answer text only -------------------------------------------


async def test_stream_yields_text_deltas_only():
    recorder = Recorder()
    script = [
        tool_round("call-1", "read", '{"path": "a.md"}', preamble="Looking. "),
        text_round("Done", "."),
    ]
    session = ChatLoopSession(
        ScriptedTransport(script), instructions="sys", tools=[], invoke=recorder.invoke
    )
    assert [delta async for delta in session.stream("go")] == ["Looking. ", "Done", "."]
    assert recorder.calls == [("read", {"path": "a.md"})]


async def test_respond_returns_what_stream_yields():
    """One loop behind both, so `AgentResult.text` is what a streaming CLI printed."""
    script = [tool_round("call-1", "read", '{"path": "a.md"}', preamble="Looking. "), text_round("Done", ".")]
    streamed = ChatLoopSession(
        ScriptedTransport([list(r) for r in script]), instructions="sys", tools=[], invoke=Recorder().invoke
    )
    answered = ChatLoopSession(
        ScriptedTransport([list(r) for r in script]), instructions="sys", tools=[], invoke=Recorder().invoke
    )
    joined = "".join([delta async for delta in streamed.stream("go")])
    assert joined == await answered.respond("go")


# -- AC-3: guided generation passes the schema through ----------------------------------


async def test_respond_structured_passes_the_schema_and_parses_the_answer():
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    transport = ScriptedTransport([text_round('{"title": ', '"ok"}')])
    session = ChatLoopSession(transport, instructions="sys", tools=[], invoke=Recorder().invoke)
    assert await session.respond_structured("go", schema) == {"title": "ok"}
    assert transport.calls[0].schema == schema


async def test_the_schema_goes_with_every_request_of_the_turn():
    """Which request is the last one is not knowable in advance (see `## 発見` on #36)."""
    schema = {"type": "object", "properties": {}}
    transport = ScriptedTransport(
        [tool_round("call-1", "read", '{"path": "a.md"}'), text_round("{}")]
    )
    session = ChatLoopSession(transport, instructions="sys", tools=[], invoke=Recorder().invoke)
    await session.respond_structured("go", schema)
    assert [call.schema for call in transport.calls] == [schema, schema]


@pytest.mark.parametrize("answer", ["not json at all", "[1, 2]"])
async def test_respond_structured_rejects_a_non_object_answer(answer: str):
    session = ChatLoopSession(
        ScriptedTransport([text_round(answer)]), instructions="sys", tools=[], invoke=Recorder().invoke
    )
    with pytest.raises(ProviderError):
        await session.respond_structured("go", {"type": "object"})


# -- AC-4: reported usage replaces the estimate ----------------------------------------


async def test_usage_is_the_last_count_the_provider_reported(meeting_ws: Path):
    agent, _, provider = scripted_agent(
        meeting_ws,
        [
            tool_round("call-1", "read", '{"path": "notes/todo.md"}') + [ChatChunk(usage=Usage(10, 2))],
            text_round("done", usage=Usage(1200, 300)),
        ],
    )
    session = agent.new_session()
    result = await session.run("summarize")

    assert result.usage == Usage(input_tokens=1200, output_tokens=300)
    usage = session.context_usage()
    assert usage.estimated is False and usage.used_tokens == 1500
    assert usage.window_tokens == 4096 and "estimated" not in usage.summary()
    assert provider.sessions[0].usage() == Usage(1200, 300)
    await session.close()


async def test_usage_stays_none_when_the_transport_counts_nothing():
    session = ChatLoopSession(
        ScriptedTransport([text_round("done")]), instructions="sys", tools=[], invoke=Recorder().invoke
    )
    await session.respond("go")
    assert session.usage() is None


# -- AC-10: the loop stops once the tool budget is spent -------------------------------


async def test_loop_stops_after_the_tool_limit(meeting_ws: Path):
    # Distinct arguments each round, so the tool budget (not the repeated-call
    # guardrail) is what ends the turn.
    agent, transport, provider = scripted_agent(
        meeting_ws,
        [tool_round(f"call-{i}", "glob", f'{{"pattern": "**/*{i}*"}}') for i in range(20)],
        config=KennelConfig(max_tool_calls=3),
    )
    result = await agent.run("keep going")

    assert result.stop_reason == "tool_limit"
    assert len(transport.calls) <= 3 + 2  # the budget, the request that exceeds it, one more
    assert [call.status for call in result.tool_calls] == ["ok", "ok", "ok", "blocked"]
    assert "limit" in (result.tool_calls[-1].error or "")
    # The abandoned request is not in the conversation: an assistant message whose tool
    # calls have no replies is not something a model can be asked to continue from.
    assert provider.sessions[0].messages[-1].role == "tool"


class BudgetedRecorder(Recorder):
    """An invoker that is not a ToolRunner and publishes its own budget."""

    def __init__(self, budget: int) -> None:
        super().__init__()
        self.budget = budget

    def budget_spent(self) -> bool:
        return len(self.calls) >= self.budget


async def test_an_explicit_limit_hit_callable_wins():
    """A non-Kennel invoker can supply the budget itself (there is no ToolRunner to read)."""
    recorder = BudgetedRecorder(budget=2)
    transport = ScriptedTransport(
        [tool_round(f"call-{i}", "read", f'{{"path": "{i}.md"}}') for i in range(5)]
    )
    session = ChatLoopSession(
        transport,
        instructions="sys",
        tools=[],
        invoke=recorder.invoke,
        limit_hit=recorder.budget_spent,
    )
    assert await session.respond("go") == ""  # the extra round asked for tools again
    assert len(transport.calls) == 3 and len(recorder.calls) == 2


async def test_close_closes_the_transport():
    transport = ScriptedTransport([text_round("done")])
    session = ChatLoopSession(transport, instructions="sys", tools=[], invoke=Recorder().invoke)
    await session.close()
    assert transport.closed
