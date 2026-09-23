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
import os
import socket
import subprocess
import sys
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
    api_key: str | None = None  # like llama-server --api-key: every route but /health needs it


#: What llama-server --api-key answers without a key or with a wrong one (measured 2026-09-23).
UNAUTHORIZED_BODY = {"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}


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

    def _authorized(self) -> bool:
        """Record the Authorization header, and answer 401 the way llama-server does."""
        header = self.headers.get("Authorization")
        self.server.auth.append((self.command, self.path, header))  # type: ignore[attr-defined]
        key = self.config.api_key
        if key is None or self.path.endswith("/health") or header == f"Bearer {key}":
            return True
        self._json(UNAUTHORIZED_BODY, 401)
        return False

    def do_GET(self) -> None:
        if not self._authorized():
            return
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
        body = self.rfile.read(length)
        if not self._authorized():
            return
        self.server.requests.append(json.loads(body or b"{}"))  # type: ignore[attr-defined]
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
    server.auth = []  # type: ignore[attr-defined]  # (method, path, Authorization header)
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


# -- the API key (issue #74) ---------------------------------------------------

KEY = "sk-test-0123456789abcdef"
KEY_ENV = "KENNEL_LLAMA_SERVER_API_KEY"


@pytest.fixture
def no_key_env(monkeypatch):
    """The developer's own key variables must not leak into these tests."""
    for name in (KEY_ENV, "LLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def llama_agent(ws: Path, server: ThreadingHTTPServer, **options: Any) -> Agent:
    """``Agent(provider="llama-server")`` configured from kennel.json, as a user would, with the
    session event log on so that what it records can be checked too."""
    config = {
        "provider": "llama-server",
        "providers": {"llama-server": {"base_url": base_url(server), "timeout": 5.0, **options}},
        "logging": {"events": True},
    }
    (ws / "kennel.json").write_text(json.dumps(config))
    return Agent(ws, tools=[])


async def test_the_key_from_the_environment_goes_on_every_request(fake_server, meeting_ws: Path, no_key_env):
    """AC-1: /props, /v1/models and the chat request all carry the Bearer header."""
    no_key_env.setenv(KEY_ENV, KEY)
    fake_server.config.api_key = KEY
    fake_server.config.replies = [answer("keyed")]
    result = await llama_agent(meeting_ws, fake_server).run("hi")
    assert result.text == "keyed"
    seen = {(method, path.split("?")[0]): header for method, path, header in fake_server.auth}
    assert set(seen) == {("GET", "/props"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions")}
    assert set(seen.values()) == {f"Bearer {KEY}"}


async def test_the_config_can_name_another_variable(fake_server, meeting_ws: Path, no_key_env):
    """AC-1: secrets.api_key names the variable (here llama-server's own LLAMA_API_KEY)."""
    no_key_env.setenv("LLAMA_API_KEY", KEY)
    fake_server.config.api_key = KEY
    fake_server.config.replies = [answer("keyed")]
    agent = llama_agent(meeting_ws, fake_server, secrets={"api_key": "LLAMA_API_KEY"})
    assert (await agent.run("hi")).text == "keyed"
    assert {header for _, _, header in fake_server.auth} == {f"Bearer {KEY}"}


async def test_without_a_key_no_header_is_sent(fake_server, meeting_ws: Path, no_key_env):
    """AC-2: nothing set, nothing sent, and an open server works as before."""
    fake_server.config.replies = [answer("open")]
    assert (await llama_agent(meeting_ws, fake_server).run("hi")).text == "open"
    assert fake_server.auth and {header for _, _, header in fake_server.auth} == {None}


@pytest.mark.parametrize(
    ("env_key", "expected"),
    [(None, "no API key was sent"), ("sk-wrong-key", "rejected the API key that was sent")],
)
async def test_a_401_is_a_fixed_provider_error(fake_server, meeting_ws: Path, state_dir: Path, no_key_env, env_key, expected):
    """AC-3: status 401, says whether a key went out and where to set it; no body, no key anywhere."""
    if env_key:
        no_key_env.setenv(KEY_ENV, env_key)
    fake_server.config.api_key = KEY
    agent = llama_agent(meeting_ws, fake_server)
    events: list[Any] = []
    agent.events.subscribe(events.append)
    with pytest.raises(ProviderError) as info:
        await agent.run("hi")
    error = info.value
    assert error.status == 401 and error.provider_error is None
    assert expected in str(error)
    assert "providers.llama-server.secrets.api_key" in str(error) and KEY_ENV in str(error)
    logs = "".join(path.read_text() for path in (state_dir / "sessions").glob("*.jsonl"))
    assert "session.failed" in logs and "session.failed" in repr(events)
    recorded = [str(error), repr(events), logs]
    for secret in ("Invalid API Key", "authentication_error", KEY, "sk-wrong-key"):
        assert all(secret not in text for text in recorded), secret


async def test_a_401_during_a_turn_is_the_same_error(fake_server, meeting_ws: Path, state_dir: Path, no_key_env):
    """AC-3: a probe that passed and a chat request refused (the key rotated) fail the turn alike."""
    no_key_env.setenv(KEY_ENV, KEY)
    agent = llama_agent(meeting_ws, fake_server)
    agent.check_availability()
    fake_server.config.api_key = "sk-rotated"
    events: list[Any] = []
    agent.events.subscribe(events.append)
    with pytest.raises(ProviderError) as info:
        await agent.run("hi")
    assert info.value.status == 401 and "rejected the API key that was sent" in str(info.value)
    failed = [event for event in events if event.type == "session.failed"]
    assert failed and failed[0].data["status"] == 401 and failed[0].data["provider_error"] is None
    logs = "".join(path.read_text() for path in (state_dir / "sessions").glob("*.jsonl"))
    assert "session.failed" in logs
    for secret in ("Invalid API Key", KEY, "sk-rotated"):
        assert secret not in repr(events) and secret not in logs


def test_the_key_is_in_no_repr(no_key_env):
    """Condition from review: the key never shows up in a repr of what holds it."""
    provider = LlamaServerProvider(base_url="http://127.0.0.1:9", api_key=KEY)
    assert KEY not in repr(provider) and KEY not in repr(provider._endpoint)
    assert KEY not in repr(vars(provider))


def test_the_cli_startup_reports_the_fixed_401_text(fake_server, meeting_ws: Path, no_key_env):
    """The CLI probes before the first turn (cli/main.py) and shows the same message."""
    no_key_env.setenv(KEY_ENV, "sk-wrong-key")
    fake_server.config.api_key = KEY
    config = {"providers": {"llama-server": {"base_url": base_url(fake_server), "timeout": 5.0}}}
    (meeting_ws / "kennel.json").write_text(json.dumps(config))
    proc = subprocess.run(
        [sys.executable, "-m", "kennel.cli.main", str(meeting_ws), "--provider", "llama-server", "-p", "hi"],
        capture_output=True,
        text=True,
        env={**os.environ, "NO_COLOR": "1"},
        timeout=60,
    )
    assert proc.returncode != 0
    assert "rejected the API key that was sent (HTTP 401)" in proc.stderr
    assert "providers.llama-server.secrets.api_key" in proc.stderr
    for secret in ("Invalid API Key", KEY, "sk-wrong-key"):
        assert secret not in proc.stdout + proc.stderr


# -- kennel doctor and the key (issue #74, AC-5) ---------------------------------


def doctor_for(ws: Path, server: ThreadingHTTPServer | None = None, **options: Any):
    from kennel.cli.doctor import render_json, render_text, run_checks

    if server is not None:
        options = {"base_url": base_url(server), "timeout": 5.0, **options}
    (ws / "kennel.json").write_text(json.dumps({"providers": {"llama-server": options}}))
    checks = run_checks(str(ws), "llama-server", user_config=None)
    return checks, render_text(checks) + render_json(checks, all(c.ok for c in checks))


def test_doctor_names_the_variable_and_whether_it_is_set(fake_server, meeting_ws: Path, no_key_env):
    no_key_env.setenv(KEY_ENV, KEY)
    fake_server.config.api_key = KEY
    checks, output = doctor_for(meeting_ws, fake_server)
    key = next(c for c in checks if c.name == "provider key api_key")
    assert key.ok and key.detail == f"llama-server api_key: {KEY_ENV} set"
    assert next(c for c in checks if c.name == "llama-server").ok  # asked with the key
    assert KEY not in output


def test_doctor_says_an_unset_key_is_optional(fake_server, meeting_ws: Path, no_key_env):
    checks, _ = doctor_for(meeting_ws, fake_server)
    key = next(c for c in checks if c.name == "provider key api_key")
    assert key.ok and key.detail == f"llama-server api_key: {KEY_ENV} not set (optional)"


@pytest.mark.parametrize("env_key", [None, "sk-wrong-key"])
def test_doctor_hints_the_variable_actually_read_on_a_401(fake_server, meeting_ws: Path, no_key_env, env_key):
    if env_key:
        no_key_env.setenv("LLAMA_API_KEY", env_key)
    fake_server.config.api_key = KEY
    checks, output = doctor_for(meeting_ws, fake_server, secrets={"api_key": "LLAMA_API_KEY"})
    server = next(c for c in checks if c.name == "llama-server")
    model = next(c for c in checks if c.name == "model")
    assert not server.ok and "HTTP 401" in server.detail
    assert "LLAMA_API_KEY" in (server.hint or "")
    assert not model.ok and model.detail.startswith("skipped:")
    for secret in ("Invalid API Key", KEY, "sk-wrong-key"):
        assert secret not in output


def test_doctor_warns_when_the_key_goes_over_plain_http_to_another_host(meeting_ws: Path, no_key_env):
    """Built without I/O: the warning comes before the (failing) probe of a closed port."""
    no_key_env.setenv(KEY_ENV, KEY)
    checks, output = doctor_for(meeting_ws, base_url="http://192.0.2.1:9", timeout=0.2)
    warning = next(c for c in checks if c.name == "llama-server key transport")
    assert warning.ok and warning.detail.startswith("warning:") and "unencrypted" in warning.detail
    assert KEY not in output


@pytest.mark.parametrize(
    ("url", "env_key"),
    [("http://127.0.0.1:9", KEY), ("https://192.0.2.1:9", KEY), ("http://192.0.2.1:9", None)],
)
def test_no_plain_http_warning_when_the_key_stays_safe(url: str, env_key: str | None, meeting_ws: Path, no_key_env):
    if env_key:
        no_key_env.setenv(KEY_ENV, env_key)
    checks, _ = doctor_for(meeting_ws, base_url=url, timeout=0.2)
    assert all(c.name != "llama-server key transport" for c in checks)
