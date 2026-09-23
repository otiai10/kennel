"""The ``web`` tool with a configured search provider (issue #61, AC-1/2/3/12)."""

import json
from pathlib import Path

import pytest

from kennel import (
    Agent,
    EventType,
    MockProvider,
    SearchProviderInfo,
    SearchResult,
    SearchSpec,
    Secret,
    ToolExecutionError,
    register_search_provider,
)
from kennel.cli.main import _header, build_agent, build_parser
from kennel.providers.mock import Text, ToolCall
from kennel.search.registry import _SPECS
from kennel.tools.web import WebSearchTool

TITLE = "Distinctive Title Zebra"
SNIPPET = "distinctive snippet okapi"
KEY_VALUE = "super-secret-key-value-42"


class FakeSearch:
    def __init__(self, count: int = 12, api_key: str | None = None) -> None:
        self.info = SearchProviderInfo("fake", "search.example", "remote")
        self.count = count
        self.api_key = api_key
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.calls.append((query, limit))
        return [SearchResult(f"{TITLE} {i}", f"https://r{i}.example", SNIPPET) for i in range(min(limit, self.count))]


@pytest.fixture
def fake_registered():
    built: list[FakeSearch] = []

    def factory(**options):
        provider = FakeSearch(**options)
        built.append(provider)
        return provider

    register_search_provider(SearchSpec("fake", factory))
    register_search_provider(SearchSpec("fakekeyed", factory, secrets={"api_key": Secret("FAKE_SEARCH_KEY")}))
    try:
        yield built
    finally:
        _SPECS.pop("fake", None)
        _SPECS.pop("fakekeyed", None)


def configure(ws: Path, data: dict) -> None:
    (ws / "kennel.json").write_text(json.dumps(data))


def web_agent(ws: Path, turns, **kw) -> tuple[Agent, MockProvider, list]:
    mock = MockProvider(turns)
    events: list = []
    agent = Agent(ws, provider=mock, tools=["read", "web"], permissions={"web": "allow"}, **kw)
    agent.events.subscribe(events.append)
    return agent, mock, events


# -- AC-1 ----------------------------------------------------------------------


async def test_configured_provider_returns_numbered_results_and_clamps(meeting_ws, fake_registered):
    configure(meeting_ws, {"search_provider": "fake"})
    turns = [[ToolCall("web", {"query": "q", "limit": 50}), ToolCall("web", {"query": "q", "limit": 0}), Text("ok")]]
    agent, mock, _ = web_agent(meeting_ws, turns)
    await agent.run("search")
    [provider] = fake_registered
    assert [limit for _, limit in provider.calls] == [10, 5]  # 50 -> 10; 0 -> the default 5
    first = mock.sessions[0].tool_results[0]
    assert first.startswith(f"1. {TITLE} 0\n   https://r0.example\n   {SNIPPET}")
    assert f"10. {TITLE} 9" in first


async def test_events_and_session_log_carry_counts_not_text(meeting_ws, fake_registered):
    configure(meeting_ws, {"search_provider": "fake", "logging": {"events": True}})
    agent, _, events = web_agent(meeting_ws, [[ToolCall("web", {"query": "q", "limit": 3}), Text("ok")]])
    session = agent.new_session()
    result = await session.run("search")
    await session.close()
    [completed] = [e for e in events if e.type is EventType.TOOL_COMPLETED]
    assert completed.data["metadata"] == {"provider": "fake", "returned": 3}
    assert result.tool_calls[0].metadata == {"provider": "fake", "returned": 3}
    everything = json.dumps([e.data for e in events]) + session.log_path.read_text(encoding="utf-8")
    assert TITLE not in everything and SNIPPET not in everything and "r0.example" not in everything


# -- AC-2 ----------------------------------------------------------------------


async def test_unconfigured_web_says_not_configured(meeting_ws):
    agent, mock, _ = web_agent(meeting_ws, [[ToolCall("web", {"query": "q"}), Text("ok")]])
    result = await agent.run("search")
    assert result.tool_calls[0].status == "error"
    assert "web search is not configured" in mock.sessions[0].tool_results[0]
    with pytest.raises(ToolExecutionError, match="web search is not configured"):
        await WebSearchTool().execute({"query": "q"}, agent.tool_context())


# -- AC-3 ----------------------------------------------------------------------


async def test_missing_key_does_not_break_start_and_names_the_variable(meeting_ws, fake_registered, monkeypatch):
    monkeypatch.delenv("FAKE_SEARCH_KEY", raising=False)
    configure(meeting_ws, {"search_provider": "fakekeyed", "logging": {"events": True}})
    agent, mock, events = web_agent(meeting_ws, [[ToolCall("web", {"query": "q"}), Text("ok")]])  # starts fine
    session = agent.new_session()
    result = await session.run("search")
    await session.close()
    assert result.stop_reason == "end_turn" and result.tool_calls[0].status == "error"
    answer = mock.sessions[0].tool_results[0]
    assert answer.startswith("Error: ") and "FAKE_SEARCH_KEY is not set" in answer
    assert "FAKE_SEARCH_KEY" in agent.web_search()
    assert not fake_registered  # never built without its key


async def test_the_key_value_appears_nowhere(meeting_ws, fake_registered, monkeypatch):
    monkeypatch.setenv("FAKE_SEARCH_KEY", KEY_VALUE)
    configure(meeting_ws, {"search_provider": "fakekeyed", "logging": {"events": True}})
    agent, _, events = web_agent(meeting_ws, [[ToolCall("web", {"query": "q"}), Text("ok")]])
    session = agent.new_session()
    result = await session.run("search")
    status = session.status()
    await session.close()
    assert fake_registered[0].api_key == KEY_VALUE  # the factory got it, under its own name
    visible = "\n".join(
        [
            json.dumps([e.data for e in events]),
            session.log_path.read_text(encoding="utf-8"),
            json.dumps(result.to_dict()),
            json.dumps(status),
            repr(agent.config),
            _header(agent),
        ]
    )
    assert KEY_VALUE not in visible


# -- AC-12 ---------------------------------------------------------------------


def test_header_and_status_show_where_searches_go(meeting_ws):
    configure(meeting_ws, {"search_provider": "searxng", "search_providers": {"searxng": {"url": "http://localhost:8888"}}})
    agent = build_agent(build_parser().parse_args([str(meeting_ws), "--provider", "mock", "--allow-web"]), None)
    header = _header(agent).splitlines()
    assert "web search: searxng → localhost:8888 → upstream engines (remote)" in header
    assert "mode: local" in header  # still the model's mode
    status = agent.new_session().status()
    assert status["web"] == "searxng → localhost:8888 → upstream engines (remote)"
    assert status["mode"] == agent.provider.info.mode == "local"


def test_no_web_line_without_web_or_without_a_provider(meeting_ws):
    configure(meeting_ws, {"search_provider": "searxng", "search_providers": {"searxng": {"url": "http://localhost:8888"}}})
    off = build_agent(build_parser().parse_args([str(meeting_ws), "--provider", "mock"]), None)
    assert "web search:" not in _header(off) and "web" not in off.new_session().status()
    (meeting_ws / "kennel.json").unlink()
    unset = build_agent(build_parser().parse_args([str(meeting_ws), "--provider", "mock", "--allow-web"]), None)
    assert "web search:" not in _header(unset) and "web" not in unset.new_session().status()
