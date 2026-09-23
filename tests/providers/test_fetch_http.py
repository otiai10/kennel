"""The fetch tool against a loopback HTTP(S) server (issue #78). Nothing leaves the machine.

Host names are made up (``fetch.test``, ``other.test``) and resolved by an injected resolver
to 127.0.0.1, so the tool is exercised exactly as for a public site except for the address
check, which ``allow_private_addresses=True`` lifts (and tests/unit/test_fetch_tool.py covers).
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kennel import Agent, Approval, KennelConfig, MockProvider
from kennel.errors import ToolExecutionError, TurnCancelledError
from kennel.providers.mock import Text, ToolCall
from kennel.tools import fetch
from kennel.tools.fetch import FetchTool

TLS = Path(__file__).parent.parent / "fixtures" / "tls"
THREAD = "kennel-fetch-http"


@dataclass
class Route:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    stall: float = 0.0


class FakeSite:
    """A loopback server answering each path from ``routes``; records what reached it."""

    def __init__(self, tls: bool = False) -> None:
        self.routes: dict[str, Route] = {}
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.connections = 0
        self.server_names: list[str | None] = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):  # noqa: N802 - http.server's naming
                site.requests.append((self.path, dict(self.headers)))
                route = site.routes.get(self.path, Route(404, b"not here"))
                deadline = time.monotonic() + route.stall
                while time.monotonic() < deadline:
                    time.sleep(0.02)
                try:
                    self.send_response(route.status)
                    for key, value in route.headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(route.body)))
                    self.end_headers()
                    self.wfile.write(route.body)
                except OSError:
                    pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def verify_request(self, request, client_address):
                site.connections += 1
                return True

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.scheme = "http"
        if tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(TLS / "fetch-test.pem", TLS / "fetch-test-key.pem")
            context.sni_callback = lambda sock, name, ctx: site.server_names.append(name)
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
            self.scheme = "https"
        self.thread = threading.Thread(target=self.httpd.serve_forever, args=(0.05,), daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def url(self, path: str, host: str = "fetch.test") -> str:
        return f"{self.scheme}://{host}:{self.port}{path}"

    def page(self, path: str, body: bytes | str, content_type: str = "text/html; charset=utf-8", **headers) -> None:
        data = body.encode() if isinstance(body, str) else body
        self.routes[path] = Route(200, data, {"Content-Type": content_type, **headers})

    def redirect(self, path: str, location: str, status: int = 302) -> None:
        self.routes[path] = Route(status, b"moved", {"Location": location, "Content-Type": "text/plain"})

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def site():
    server = FakeSite()
    yield server
    server.close()


@pytest.fixture
def tls_site(monkeypatch):
    server = FakeSite(tls=True)
    monkeypatch.setattr(fetch, "_ssl_context", lambda: ssl.create_default_context(cafile=str(TLS / "ca.pem")))
    yield server
    server.close()


@pytest.fixture
def no_system_dns(monkeypatch):
    """Any name resolution outside the injected resolver fails the test (no re-resolving)."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"resolved outside the injected resolver: {args[:2]}")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)


class Resolver:
    def __init__(self, answer: list[str] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.answer = answer or ["127.0.0.1"]

    def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append((host, port))
        return self.answer


def local_tool(resolver=None, **kw) -> FetchTool:
    return FetchTool(allow_private_addresses=True, resolver=resolver or Resolver(), **kw)


async def threads_gone() -> bool:
    for _ in range(100):
        if not [t for t in threading.enumerate() if t.name == THREAD]:
            return True
        await asyncio.sleep(0.02)
    return False


# -- content (AC-1, AC-5) --------------------------------------------------------


async def test_html_comes_back_as_text(site, ctx, no_system_dns):
    site.page("/page", "<html><head><script>x()</script><style>b{}</style></head><body><p>Hello <b>there</b></p></body></html>")
    result = await local_tool().execute({"url": site.url("/page")}, ctx)
    assert result.content == "Hello there"
    assert result.metadata == {"host": "fetch.test", "status": 200, "bytes": len(site.routes["/page"].body), "content_type": "text/html"}


@pytest.mark.parametrize("content_type", ["text/plain", "application/json", "application/ld+json"])
async def test_text_and_json_come_back_as_they_are(site, ctx, content_type):
    body = json.dumps({"a": "<b>1</b>"}) if "json" in content_type else "line <one>\nline two"
    site.page("/doc", body, content_type)
    assert (await local_tool().execute({"url": site.url("/doc")}, ctx)).content == body


async def test_the_charset_is_honoured(site, ctx):
    site.page("/sjis", "日本語".encode("shift_jis"), "text/plain; charset=shift_jis")
    assert (await local_tool().execute({"url": site.url("/sjis")}, ctx)).content == "日本語"


@pytest.mark.parametrize("content_type", ["image/png", "application/pdf", "application/octet-stream"])
async def test_other_types_are_an_error_without_the_body(site, ctx, content_type):
    site.page("/bin", b"SECRET-BODY", content_type)
    with pytest.raises(ToolExecutionError) as info:
        await local_tool().execute({"url": site.url("/bin")}, ctx)
    assert content_type in str(info.value) and "SECRET-BODY" not in str(info.value)


async def test_a_compressed_body_is_an_error(site, ctx):
    """No Accept-Encoding is sent; a server that compresses anyway is refused, not decoded as garbage."""
    site.page("/gz", b"\x1f\x8b\x08SECRET-BODY", "text/html", **{"Content-Encoding": "gzip"})
    with pytest.raises(ToolExecutionError, match="gzip") as info:
        await local_tool().execute({"url": site.url("/gz")}, ctx)
    assert "SECRET-BODY" not in str(info.value)
    assert site.requests[0][1].get("Accept-Encoding", "identity") == "identity"  # nothing compressed asked for


async def test_an_error_status_carries_only_the_status(site, ctx):
    site.routes["/secret?token=1"] = Route(500, b"SECRET-BODY", {"Content-Type": "text/plain"})
    with pytest.raises(ToolExecutionError) as info:
        await local_tool().execute({"url": site.url("/secret?token=1")}, ctx)
    assert "500" in str(info.value) and "SECRET-BODY" not in str(info.value) and "token" not in str(info.value)


# -- limits (AC-4) -----------------------------------------------------------------


async def test_reading_stops_at_the_byte_limit(site, ctx):
    site.page("/big", "x" * 50_000, "text/plain")
    with pytest.raises(ToolExecutionError, match="more than 1000 bytes") as info:
        await local_tool(max_bytes=1000).execute({"url": site.url("/big")}, ctx)
    assert "xxxx" not in str(info.value)


async def test_a_slow_server_hits_the_deadline(site, ctx):
    site.routes["/slow"] = Route(200, b"late", {"Content-Type": "text/plain"}, stall=3.0)
    started = time.perf_counter()
    with pytest.raises(ToolExecutionError, match="did not answer within 0.5 s"):
        await local_tool(timeout=0.5).execute({"url": site.url("/slow")}, ctx)
    assert time.perf_counter() - started < 2.0


# -- pinning and TLS (AC-3) ---------------------------------------------------------


async def test_it_connects_to_the_checked_address_without_resolving_again(site, ctx, no_system_dns):
    site.page("/p", "pinned", "text/plain")
    resolver = Resolver()
    result = await local_tool(resolver).execute({"url": site.url("/p")}, ctx)
    assert result.content == "pinned"
    assert resolver.calls == [("fetch.test", site.port)]
    assert site.requests[0][1]["Host"] == f"fetch.test:{site.port}"  # the name, not the address


async def test_https_verifies_the_certificate_against_the_name(tls_site, ctx, no_system_dns):
    tls_site.page("/secure", "<p>over tls</p>")
    result = await local_tool().execute({"url": tls_site.url("/secure")}, ctx)
    assert result.content == "over tls"
    assert tls_site.server_names == ["fetch.test"]  # SNI carries the name
    assert tls_site.requests[0][1]["Host"] == f"fetch.test:{tls_site.port}"


async def test_https_refuses_a_certificate_for_another_name(tls_site, ctx, no_system_dns):
    tls_site.page("/secure", "<p>over tls</p>")
    with pytest.raises(ToolExecutionError, match="certificate"):
        await local_tool().execute({"url": tls_site.url("/secure", host="other.test")}, ctx)
    assert tls_site.requests == []


# -- redirects (AC-7) ------------------------------------------------------------------


async def test_a_relative_redirect_on_the_same_host_is_followed_and_checked_again(site, ctx):
    site.redirect("/dir/old", "../new?x=1")
    site.page("/new?x=1", "arrived", "text/plain")
    resolver = Resolver()
    result = await local_tool(resolver).execute({"url": site.url("/dir/old")}, ctx)
    assert result.content == "arrived"
    assert [path for path, _ in site.requests] == ["/dir/old", "/new?x=1"]
    assert len(resolver.calls) == 2  # every hop resolves and checks again


async def test_five_redirects_are_followed_and_the_sixth_is_an_error(site, ctx):
    for i in range(6):
        site.redirect(f"/r{i}", f"/r{i + 1}")
    site.page("/r5", "five", "text/plain")
    assert (await local_tool().execute({"url": site.url("/r0")}, ctx)).content == "five"
    site.redirect("/r5", "/r6")
    site.page("/r6", "six", "text/plain")
    with pytest.raises(ToolExecutionError, match="more than 5 times"):
        await local_tool().execute({"url": site.url("/r0")}, ctx)


async def test_a_redirect_to_a_private_address_is_refused(site, ctx, monkeypatch):
    """The first hop passes the check (127.0.0.1 let through here); the second resolves to 10.x."""
    real = fetch._address_problem
    monkeypatch.setattr(fetch, "_address_problem", lambda a: None if a == "127.0.0.1" else real(a))
    answers = iter([["127.0.0.1"], ["10.0.0.7"]])
    site.redirect("/start", "/inside")
    tool = FetchTool(resolver=lambda host, port: next(answers))
    with pytest.raises(ToolExecutionError, match="private address") as info:
        await tool.execute({"url": site.url("/start")}, ctx)
    assert "10.0.0.7" not in str(info.value)
    assert [path for path, _ in site.requests] == ["/start"]


@pytest.mark.parametrize(
    "location",
    ["http://other.test:{port}/x?k=v", "http://fetch.test:1/x", "https://fetch.test:{port}/x"],
    ids=["another host", "another port", "http to https on another port"],
)
async def test_other_destinations_are_returned_not_followed(site, ctx, location):
    target = location.format(port=site.port)
    site.redirect("/go", target, status=301)
    result = await local_tool().execute({"url": site.url("/go")}, ctx)
    assert target in result.content and "call fetch again" in result.content
    assert result.metadata["status"] == 301
    assert [path for path, _ in site.requests] == ["/go"]


async def test_https_to_http_is_not_followed(tls_site, ctx):
    target = f"http://fetch.test:{tls_site.port}/plain"
    tls_site.redirect("/secure", target)
    result = await local_tool().execute({"url": tls_site.url("/secure")}, ctx)
    assert target in result.content and "call fetch again" in result.content


# -- what is recorded, what is asked (AC-6) -----------------------------------------------


async def test_events_and_the_log_carry_neither_the_query_nor_the_page(site, meeting_ws):
    site.page("/doc?token=SECRET-QUERY", "<p>PAGE-TEXT</p>")
    url = site.url("/doc?token=SECRET-QUERY#FRAGMENT")
    asked = []

    def prompter(request):
        asked.append(request)
        return Approval.ONCE

    agent = Agent(
        meeting_ws,
        provider=MockProvider([[ToolCall("fetch", {"url": url}), Text("done")]]),
        tools=[local_tool()],
        permissions={"fetch": "ask"},
        prompter=prompter,
        config=KennelConfig(log_events=True),
    )
    events = []
    agent.events.subscribe(events.append)
    session = agent.new_session()
    result = await session.run("read it")
    await session.close()
    [call] = result.tool_calls
    assert call.status == "ok", call.error
    assert call.arguments == {"url": url}  # the SDK result keeps the whole call
    assert call.metadata == {"host": "fetch.test", "status": 200, "bytes": 16, "content_type": "text/html"}
    assert asked[0].details == url.split("#")[0]  # the prompt shows the query (the fragment is never sent)
    assert asked[0].summary == f"Fetch {site.url('/doc')}"
    recorded = json.dumps([e.data for e in events]) + session.log_path.read_text(encoding="utf-8")
    assert "SECRET-QUERY" not in recorded and "FRAGMENT" not in recorded and "PAGE-TEXT" not in recorded
    assert f"Fetch {site.url('/doc')}" in recorded


# -- grants per host (AC-8) ------------------------------------------------------------------


async def test_a_session_grant_covers_one_host(site, meeting_ws):
    site.page("/a", "a", "text/plain")
    asked = []

    def prompter(request):
        asked.append(request.scope)
        return Approval.SESSION

    urls = [site.url("/a", host="docs.python.org"), site.url("/a", host="docs.python.org"), site.url("/a", host="other.test")]
    agent = Agent(
        meeting_ws,
        provider=MockProvider([[*(ToolCall("fetch", {"url": u}) for u in urls), Text("done")]]),
        tools=[local_tool()],
        permissions={"fetch": "ask"},
        prompter=prompter,
    )
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["ok", "ok", "ok"]
    assert asked == ["docs.python.org", "other.test"]  # the second call went through on the grant


# -- interrupting (AC-4) -----------------------------------------------------------------------


def interruptible(meeting_ws, tool: FetchTool, url: str) -> Agent:
    mock = MockProvider([[ToolCall("fetch", {"url": url}), Text("never")], ["after"]])
    return Agent(meeting_ws, provider=mock, tools=[tool], permissions={"fetch": "allow"})


async def interrupt_when(session, task, ready) -> None:
    for _ in range(150):
        if ready():
            break
        await asyncio.sleep(0.02)
    assert ready()
    started = time.perf_counter()
    session.interrupt()
    with pytest.raises(TurnCancelledError):
        await asyncio.wait_for(task, 2.0)
    assert time.perf_counter() - started < 2.0
    assert (await session.run("again")).stop_reason == "end_turn"  # the session survived
    await session.close()


async def test_interrupt_while_resolving_sends_nothing(site, meeting_ws):
    site.page("/p", "never", "text/plain")
    resolving, release = threading.Event(), threading.Event()

    def stuck_resolver(host, port):  # getaddrinfo cannot be aborted: the worker is abandoned
        resolving.set()
        release.wait(10)
        return ["127.0.0.1"]

    agent = interruptible(meeting_ws, local_tool(stuck_resolver), site.url("/p"))
    session = agent.new_session()
    await interrupt_when(session, asyncio.create_task(session.run("go")), resolving.is_set)
    release.set()  # the name "resolves" after the caller left
    assert await threads_gone()
    assert site.connections == 0 and site.requests == []


async def test_interrupt_while_connecting_sends_nothing(site, meeting_ws, monkeypatch):
    """A connect that never completes (an unanswered SYN): the poll loop sees the cancel."""
    site.page("/p", "never", "text/plain")
    connecting = threading.Event()

    def never_writable(sock, seconds):
        connecting.set()
        time.sleep(seconds)
        return False

    monkeypatch.setattr(fetch, "_wait_writable", never_writable)
    agent = interruptible(meeting_ws, local_tool(), site.url("/p"))
    session = agent.new_session()
    await interrupt_when(session, asyncio.create_task(session.run("go")), connecting.is_set)
    assert await threads_gone()
    assert site.requests == []


async def test_interrupt_while_reading_stops_the_request(site, meeting_ws):
    site.routes["/slow"] = Route(200, b"late", {"Content-Type": "text/plain"}, stall=5.0)
    agent = interruptible(meeting_ws, local_tool(timeout=30), site.url("/slow"))
    session = agent.new_session()
    await interrupt_when(session, asyncio.create_task(session.run("go")), lambda: bool(site.requests))
    assert await threads_gone()
