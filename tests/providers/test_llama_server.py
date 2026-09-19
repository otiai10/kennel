"""LlamaServerProvider against a fake llama-server (issue #36). No model needed.

The fake replays the wire shapes measured on llama-server 0.4.1 (build b10964,
Qwen3-4B-GGUF Q4_K_M, ``-c 32768 --jinja``), including the ``content: null`` first
chunk, ``reasoning_content`` deltas, the trailing ``choices: []`` usage chunk and
the ``exceed_context_size_error`` body. Real-model coverage lives in
``tests/integration/test_llama.py`` behind the ``llama`` marker.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kennel import (
    Agent,
    ContextLimitError,
    ModelUnavailableError,
    ProviderError,
    TurnCancelledError,
    Usage,
)
from kennel.providers.llama_server import LlamaServerProvider, llama_server_doctor_checks
from kennel.providers.registry import create, names

# -- the fake server ---------------------------------------------------------


@dataclass
class Reply:
    """One scripted answer to POST /v1/chat/completions."""

    events: list[dict[str, Any]] = field(default_factory=list)  # streamed as data: lines
    status: int = 200
    body: dict[str, Any] | None = None  # the JSON body when status != 200
    stall_seconds: float = 0.0  # held open before the first event, to be interrupted


@dataclass
class FakeConfig:
    n_ctx: int | None = 32768
    model_name: str | None = "fake/Model-GGUF:Q4_K_M"
    replies: list[Reply] = field(default_factory=lambda: [Reply()])


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.0: the client reads the stream until the connection closes, which is
    # also what a fresh connection per request means for the provider.
    protocol_version = "HTTP/1.0"

    def log_message(self, *args: Any) -> None:
        return None

    @property
    def config(self) -> FakeConfig:
        return self.server.config  # type: ignore[attr-defined]

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        settings: dict[str, Any] = {}
        if self.config.n_ctx is not None:
            settings["n_ctx"] = self.config.n_ctx
        if self.path.endswith("/props"):
            self._json({"default_generation_settings": settings, "model_path": "/tmp/fake.gguf"})
        elif self.path.endswith("/v1/models"):
            models = [{"name": self.config.model_name}] if self.config.model_name else []
            self._json({"models": models, "object": "list"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.server.requests.append(json.loads(self.rfile.read(length) or b"{}"))  # type: ignore[attr-defined]
        replies = self.config.replies
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if reply.status != 200:
            self._json(reply.body or {}, reply.status)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            if reply.stall_seconds:
                time.sleep(reply.stall_seconds)
            for event in reply.events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass  # the client went away (an interrupted turn)


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.config = FakeConfig()  # type: ignore[attr-defined]
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def base_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def provider_for(server: ThreadingHTTPServer, **options: Any) -> LlamaServerProvider:
    return LlamaServerProvider(base_url=base_url(server), timeout=5.0, **options)


def closed_port() -> int:
    """A port nothing listens on (bound then released), for the refused-connection case."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# -- event builders, shaped like the real stream ------------------------------


def delta_event(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "finish_reason": None, "delta": delta}], "object": "chat.completion.chunk"}


def finish_event(reason: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "finish_reason": reason, "delta": {}}]}


def usage_event(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timings": {"predicted_n": completion_tokens},
    }


def answer(text: str, *, prompt_tokens: int = 158, completion_tokens: int = 20) -> Reply:
    """A plain answer, preceded by the real server's `content: null` opener."""
    events = [delta_event(role="assistant", content=None)]
    events += [delta_event(content=piece) for piece in text]
    events += [finish_event("stop"), usage_event(prompt_tokens, completion_tokens)]
    return Reply(events=events)


def tool_call(call_id: str, name: str, arguments: str) -> Reply:
    """A tool call, split into fragments exactly as the real server splits it."""
    events = [
        delta_event(role="assistant", content=None),
        delta_event(tool_calls=[{"index": 0, "id": call_id, "type": "function", "function": {"name": name, "arguments": arguments[:1]}}]),
    ]
    events += [
        delta_event(tool_calls=[{"index": 0, "function": {"arguments": piece}}])
        for piece in arguments[1:]
    ]
    events += [finish_event("tool_calls"), usage_event(170, 15)]
    return Reply(events=events)


# -- AC-5: what the provider learns from /props and /v1/models ----------------


def test_check_availability_reads_props_and_models(fake_server):
    provider = provider_for(fake_server)
    provider.check_availability()
    assert provider.info.name == "llama-server"
    assert provider.info.context_window_tokens == 32768
    assert provider.info.model == "fake/Model-GGUF:Q4_K_M"
    assert provider.info.mode == "local"


def test_a_configured_model_name_wins_over_the_reported_one(fake_server):
    provider = provider_for(fake_server, model="my/own:tag")
    provider.check_availability()
    assert provider.info.model == "my/own:tag"


def test_a_server_that_declares_no_window_leaves_it_none(fake_server):
    fake_server.config.n_ctx = None
    fake_server.config.model_name = None
    provider = provider_for(fake_server)
    provider.check_availability()
    assert provider.info.context_window_tokens is None
    assert provider.base_url in provider.info.model  # no name to report: say where it is


@pytest.mark.parametrize(
    ("url", "mode"),
    [
        ("http://127.0.0.1:8080", "local"),
        ("http://127.0.0.2:8080", "local"),
        ("http://localhost:8080", "local"),
        ("http://[::1]:8080", "local"),
        ("http://10.0.0.5:8080", "remote"),
        ("https://models.example.com", "remote"),
    ],
)
def test_the_base_url_host_decides_the_mode(url: str, mode: str):
    """Principle 1: only loopback may call itself local, and it is declared without any I/O."""
    assert LlamaServerProvider(base_url=url).info.mode == mode


# -- AC-6: error mapping ------------------------------------------------------


def test_an_unreachable_server_is_unavailable():
    provider = LlamaServerProvider(base_url=f"http://127.0.0.1:{closed_port()}", timeout=2.0)
    with pytest.raises(ModelUnavailableError) as exc:
        provider.check_availability()
    assert "llama-server" in str(exc.value) and "--jinja" in str(exc.value)  # how to start one


async def test_a_full_window_becomes_a_context_limit_error(fake_server, meeting_ws: Path):
    fake_server.config.replies = [
        Reply(
            status=400,
            body={
                "error": {
                    "code": 400,
                    "type": "exceed_context_size_error",
                    "message": "request (40009 tokens) exceeds the available context size (32768 tokens)",
                    "n_prompt_tokens": 40009,
                    "n_ctx": 32768,
                }
            },
        )
    ]
    agent = Agent(meeting_ws, provider=provider_for(fake_server), tools=[])
    with pytest.raises(ContextLimitError):
        await agent.run("far too much")


async def test_a_server_error_is_a_provider_error(fake_server, meeting_ws: Path):
    fake_server.config.replies = [Reply(status=500, body={"error": {"message": "slot unavailable"}})]
    agent = Agent(meeting_ws, provider=provider_for(fake_server), tools=[])
    with pytest.raises(ProviderError) as exc:
        await agent.run("hi")
    assert "500" in str(exc.value) and "slot unavailable" in str(exc.value)


async def test_the_compact_and_retry_path_still_applies(fake_server, meeting_ws: Path):
    """A window overflow is retried once after compacting, as with any provider."""
    fake_server.config.replies = [
        answer("first"),
        Reply(status=400, body={"error": {"type": "exceed_context_size_error", "message": "too big"}}),
        answer("second"),
    ]
    agent = Agent(meeting_ws, provider=provider_for(fake_server), tools=[])
    session = agent.new_session()
    assert (await session.run("one")).text == "first"
    result = await session.run("two")
    assert result.text == "second" and result.compactions == 1
    await session.close()


# -- AC-4 / AC-11: what the stream carries and what it leaves out -------------


async def test_reasoning_and_usage_chunks_are_handled(fake_server, meeting_ws: Path):
    events = [delta_event(role="assistant", content=None)]
    events += [delta_event(reasoning_content=thought) for thought in ("Hmm", ", let me think")]
    events += [delta_event(content="The "), delta_event(content="answer.")]
    events += [finish_event("stop"), usage_event(1200, 300)]
    fake_server.config.replies = [Reply(events=events)]

    provider = provider_for(fake_server)
    agent = Agent(meeting_ws, provider=provider, tools=[])
    session = agent.new_session()
    deltas: list[str] = []
    result = await session.run("think", on_delta=deltas.append)

    assert deltas == ["The ", "answer."]  # no reasoning, no empty chunks
    assert result.text == "The answer."
    assert result.usage == Usage(input_tokens=1200, output_tokens=300)
    usage = session.context_usage()
    assert usage.estimated is False and usage.used_tokens == 1500
    assert usage.window_tokens == 32768
    await session.close()


async def test_a_tool_call_round_trips_over_http(fake_server, meeting_ws: Path):
    fake_server.config.replies = [
        tool_call("call-abc", "read", '{"path": "notes/todo.md"}'),
        answer("Read it."),
    ]
    agent = Agent(meeting_ws, provider=provider_for(fake_server), tools=["read"])
    result = await agent.run("what is on the list?")

    assert result.text == "Read it."
    assert [call.name for call in result.tool_calls] == ["read"]
    assert result.tool_calls[0].status == "ok"
    first, second = fake_server.requests
    assert first["stream"] is True and first["stream_options"] == {"include_usage": True}
    assert [tool["function"]["name"] for tool in first["tools"]] == ["read"]
    assert first["model"] == "fake/Model-GGUF:Q4_K_M"
    assert "tool_choice" not in first  # it leaks raw <tool_call> text; never used
    reply = second["messages"][-1]
    assert reply["role"] == "tool" and reply["tool_call_id"] == "call-abc"


async def test_extra_body_is_merged_into_every_request(fake_server, meeting_ws: Path):
    provider = provider_for(fake_server, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    agent = Agent(meeting_ws, provider=provider, tools=[])
    await agent.run("hi")
    assert fake_server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_guided_generation_sends_a_json_schema(fake_server, meeting_ws: Path):
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    fake_server.config.replies = [answer('{"title": "ok"}')]
    agent = Agent(meeting_ws, provider=provider_for(fake_server), tools=[])
    result = await agent.run("give me a title", schema=schema)

    assert result.structured_output == {"title": "ok"}
    response_format = fake_server.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema


# -- AC-7: interrupting a turn that is waiting on HTTP ------------------------


async def test_interrupt_during_the_http_wait(fake_server, meeting_ws: Path):
    fake_server.config.replies = [Reply(events=answer("never").events, stall_seconds=5.0), answer("after")]
    agent = Agent(meeting_ws, provider=provider_for(fake_server), tools=[])
    session = agent.new_session()

    task = asyncio.create_task(session.run("wait for it"))
    await asyncio.sleep(0.1)
    started = time.perf_counter()
    session.interrupt()
    with pytest.raises(TurnCancelledError):
        await asyncio.wait_for(task, 2.0)
    assert time.perf_counter() - started < 2.0

    assert (await session.run("again")).text == "after"  # the session survived
    await session.close()
    await asyncio.sleep(0.1)
    assert not [t for t in threading.enumerate() if t.name == "kennel-llama-http"]


# -- the registry and doctor wiring -------------------------------------------


def test_the_provider_is_registered_by_name():
    assert "llama-server" in names()
    provider = create("llama-server", base_url="http://127.0.0.1:9999")
    assert isinstance(provider, LlamaServerProvider) and provider.base_url == "http://127.0.0.1:9999"


def test_doctor_checks_report_the_server(fake_server):
    checks = llama_server_doctor_checks(base_url=base_url(fake_server), timeout=5.0)
    assert [check.ok for check in checks] == [True, True]
    assert "mode: local" in checks[0].detail
    assert "fake/Model-GGUF:Q4_K_M" in checks[1].detail and "32768" in checks[1].detail


def test_doctor_checks_explain_an_unreachable_server():
    checks = llama_server_doctor_checks(base_url=f"http://127.0.0.1:{closed_port()}", timeout=2.0)
    assert [check.ok for check in checks] == [False, False]
    assert "llama-server" in (checks[0].hint or "")
