"""The built-in search providers against a fake HTTP service (issue #61, AC-9/10/11).

The conformance tests run for every built-in provider; the rest pin each service's own
failure modes. Nothing here leaves the machine: the service is a loopback fake.
"""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from kennel import Agent, MockProvider, SearchError, SearchProvider, TurnCancelledError
from kennel.providers.mock import Text, ToolCall
from kennel.search.brave import BraveProvider
from kennel.search.searxng import SearxngProvider
from kennel.tools.web import WebSearchTool

FAKE_KEY = "brv-secret-value-123"
NOT_HELPFUL = "Repeating this search will not help"


def searxng(url: str, **kw) -> SearxngProvider:
    return SearxngProvider(url, **kw)


def brave(url: str, **kw) -> BraveProvider:
    return BraveProvider(FAKE_KEY, endpoint=url, **kw)


# One entry per built-in provider: how to build it and what its "empty" and "hits" look like.
PROVIDERS = {
    "searxng": (
        searxng,
        {"results": [], "unresponsive_engines": []},
        {"results": [{"title": "T1", "url": "https://a.example", "content": "S1"}, {"title": "T2", "url": "https://b.example", "content": "S2"}]},
    ),
    "brave": (
        brave,
        {"type": "search"},
        {"web": {"results": [{"title": "T1", "url": "https://a.example", "description": "S1"}, {"title": "T2", "url": "https://b.example", "description": "S2"}]}},
    ),
}


@pytest.fixture(params=sorted(PROVIDERS))
def kind(request):
    return request.param


# -- conformance: every provider keeps the same promises (AC-9) ----------------


def test_conformance_is_a_search_provider(kind, search_server):
    build, _, _ = PROVIDERS[kind]
    provider = build(search_server.url)
    assert isinstance(provider, SearchProvider)
    assert provider.info.name == kind and provider.info.mode == "remote"


async def test_conformance_results_are_mapped_and_limited(kind, search_server):
    build, _, hits = PROVIDERS[kind]
    search_server.reply(200, hits)
    results = await build(search_server.url).search("q", limit=1)
    assert [(r.title, r.url, r.snippet) for r in results] == [("T1", "https://a.example", "S1")]


async def test_conformance_nothing_found_is_an_empty_list(kind, search_server):
    build, empty, _ = PROVIDERS[kind]
    search_server.reply(200, empty)
    assert await build(search_server.url).search("q") == []


async def test_conformance_unreachable_is_a_search_error(kind):
    build, _, _ = PROVIDERS[kind]
    import socket

    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(SearchError) as info:
        await build(f"http://127.0.0.1:{port}").search("q")
    assert info.value.retryable and "Retrying later may help" in str(info.value)


async def test_conformance_timeout_is_a_search_error(kind, search_server):
    """AC-11: a service that does not answer fails at the deadline."""
    build, _, hits = PROVIDERS[kind]
    search_server.reply(200, hits)
    search_server.stall = 3.0
    started = time.perf_counter()
    with pytest.raises(SearchError, match="did not answer within 0.3 s"):
        await build(search_server.url, timeout=0.3).search("q")
    assert time.perf_counter() - started < 2.0


@pytest.mark.parametrize("send_length", [True, False])
async def test_conformance_oversized_body_stops_at_the_limit(kind, search_server, send_length):
    """AC-11: reading stops at the limit (with or without a Content-Length to go by)."""
    build, _, _ = PROVIDERS[kind]
    search_server.reply(200, b"x" * 65536)
    search_server.body_chunks = 200  # 12.5 MiB offered
    search_server.send_length = send_length
    with pytest.raises(SearchError, match="sent more than 100000 bytes") as info:
        await build(search_server.url, max_response_bytes=100_000).search("q")
    assert NOT_HELPFUL in str(info.value)
    assert search_server.disconnected.wait(3.0)  # the client left long before 12.5 MiB


async def test_conformance_a_permanent_failure_reaches_the_model_as_error_text(kind, search_server, meeting_ws: Path):
    """AC-9: 'repeating will not help' and the remedy are in the model's ``Error:`` text."""
    build, _, _ = PROVIDERS[kind]
    search_server.reply(403, {"error": {"code": "X"}})
    mock = MockProvider([[ToolCall("web", {"query": "q"}), Text("done")]])
    agent = Agent(meeting_ws, provider=mock, tools=[WebSearchTool(build(search_server.url))], permissions={"web": "allow"})
    result = await agent.run("search")
    [call] = result.tool_calls
    assert call.status == "error" and NOT_HELPFUL in call.error
    sent = mock.sessions[0].tool_results[0]
    assert sent.startswith("Error: ") and NOT_HELPFUL in sent


# -- interrupting a search (AC-11) ---------------------------------------------


async def test_interrupt_stops_the_request(search_server, meeting_ws: Path):
    search_server.reply(200, {"results": []})
    search_server.stall = 5.0
    mock = MockProvider([[ToolCall("web", {"query": "q"}), Text("never")], ["after"]])
    agent = Agent(
        meeting_ws, provider=mock, tools=[WebSearchTool(searxng(search_server.url, timeout=30))], permissions={"web": "allow"}
    )
    session = agent.new_session()
    task = asyncio.create_task(session.run("search"))
    for _ in range(100):
        if search_server.requests:
            break
        await asyncio.sleep(0.02)
    assert search_server.requests
    started = time.perf_counter()
    session.interrupt()
    with pytest.raises(TurnCancelledError):
        await asyncio.wait_for(task, 2.0)
    assert time.perf_counter() - started < 2.0
    assert (await session.run("again")).stop_reason == "end_turn"  # the session survived
    await session.close()
    for _ in range(50):
        if not [t for t in threading.enumerate() if t.name == "kennel-search-http"]:
            break
        await asyncio.sleep(0.02)
    assert not [t for t in threading.enumerate() if t.name == "kennel-search-http"]  # the request stopped


# -- SearXNG (AC-10) -----------------------------------------------------------


async def test_searxng_json_disabled_is_explained(search_server):
    search_server.reply(403, b"<html>Forbidden</html>")
    with pytest.raises(SearchError) as info:
        await searxng(search_server.url).search("q")
    message = str(info.value)
    assert "HTTP 403" in message and "search.formats" in message and NOT_HELPFUL in message


async def test_searxng_invalid_json_is_a_search_error(search_server):
    search_server.reply(200, b"<html>not json</html>")
    with pytest.raises(SearchError, match="not its JSON format") as info:
        await searxng(search_server.url).search("q")
    assert NOT_HELPFUL in str(info.value) and "search_providers.searxng.url" in str(info.value)


async def test_searxng_all_engines_unresponsive_is_a_failure_not_nothing(search_server):
    search_server.reply(200, {"results": [], "unresponsive_engines": [["google", "CAPTCHA"], ["bing", "timeout"]]})
    with pytest.raises(SearchError) as info:
        await searxng(search_server.url).search("q")
    message = str(info.value)
    assert "google: CAPTCHA" in message and "bing: timeout" in message and NOT_HELPFUL in message


async def test_searxng_partial_results_are_kept(search_server):
    search_server.reply(
        200,
        {"results": [{"title": "T", "url": "https://a.example", "content": "S"}], "unresponsive_engines": [["google", "CAPTCHA"]]},
    )
    assert [r.url for r in await searxng(search_server.url).search("q")] == ["https://a.example"]


async def test_searxng_sends_the_json_query(search_server):
    await searxng(search_server.url + "/sx/").search("kennel test")
    path, _ = search_server.requests[0]
    assert path == "/sx/search?q=kennel+test&format=json"


def test_searxng_info_names_the_relay():
    info = searxng("http://localhost:8888").info
    assert info.destination == "localhost:8888 → upstream engines" and info.mode == "remote"


# -- Brave (AC-10) -------------------------------------------------------------


async def test_brave_sends_the_key_in_the_header_only(search_server):
    search_server.reply(200, {"web": {"results": []}})
    await brave(search_server.url).search("kennel", limit=3)
    path, headers = search_server.requests[0]
    assert path == "/res/v1/web/search?q=kennel&count=3"
    assert headers["X-Subscription-Token"] == FAKE_KEY
    assert FAKE_KEY not in path and FAKE_KEY not in repr(brave(search_server.url))


@pytest.mark.parametrize(
    "status,body,headers",
    [
        (401, {"type": "ErrorResponse", "error": {"status": 401}}, {}),
        (422, {"type": "ErrorResponse", "error": {"code": "SUBSCRIPTION_TOKEN_INVALID", "status": 422}}, {}),
    ],
)
async def test_brave_auth_failure(search_server, status, body, headers):
    search_server.reply(status, body, headers)
    with pytest.raises(SearchError, match="rejected the API key") as info:
        await brave(search_server.url).search("q")
    assert not info.value.retryable and NOT_HELPFUL in str(info.value) and "plan is active" in str(info.value)
    assert FAKE_KEY not in str(info.value)


@pytest.mark.parametrize(
    "status,body,headers",
    [
        (429, {"error": {"code": "RATE_LIMITED"}}, {"X-RateLimit-Remaining": "0, 0"}),
        (429, {"error": {"code": "QUOTA_LIMITED"}}, {}),
        (402, {}, {}),
    ],
)
async def test_brave_quota_exceeded(search_server, status, body, headers):
    search_server.reply(status, body, headers)
    with pytest.raises(SearchError, match="quota exceeded") as info:
        await brave(search_server.url).search("q")
    assert not info.value.retryable and NOT_HELPFUL in str(info.value) and "dashboard" in str(info.value)


async def test_brave_rate_limited(search_server):
    search_server.reply(429, {"error": {"code": "RATE_LIMITED"}}, {"X-RateLimit-Remaining": "0, 900"})
    with pytest.raises(SearchError, match="rate limit") as info:
        await brave(search_server.url).search("q")
    assert info.value.retryable and "Retrying later may help" in str(info.value) and "1 request per second" in str(info.value)


async def test_brave_strips_markup_from_snippets(search_server):
    search_server.reply(
        200, {"web": {"results": [{"title": "A &amp; B", "url": "https://a.example", "description": "the <strong>word</strong>"}]}}
    )
    [result] = await brave(search_server.url).search("q")
    assert (result.title, result.snippet) == ("A & B", "the word")
