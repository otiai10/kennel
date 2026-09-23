"""Choosing a search provider by name, from config, in the SDK and the CLI (issue #61, AC-4/8/13)."""

import json
from pathlib import Path

import pytest

import kennel
from kennel import (
    Agent,
    ConfigurationError,
    Decision,
    MockProvider,
    SearchError,
    SearchProvider,
    SearchProviderInfo,
    SearchResult,
    SearchSpec,
    Secret,
    ToolExecutionError,
    register_search_provider,
)
from kennel.cli.main import build_agent, build_parser
from kennel.config import KennelConfig, apply_config, load_config
from kennel.search import registry
from kennel.search.brave import BraveProvider
from kennel.search.registry import _SPECS
from kennel.tools.web import WebSearchTool

README = Path(__file__).resolve().parents[2] / "README.md"


class Named:
    def __init__(self, name: str = "fake") -> None:
        self.info = SearchProviderInfo(name, "search.example")

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [SearchResult(self.info.name, "https://x.example", "s")]


@pytest.fixture
def fake_registered():
    register_search_provider(SearchSpec("fake", lambda **_: Named("fake")))
    try:
        yield
    finally:
        _SPECS.pop("fake", None)


def configure(ws: Path, data: dict) -> None:
    (ws / "kennel.json").write_text(json.dumps(data))


def cli_agent(ws: Path, *flags: str) -> Agent:
    return build_agent(build_parser().parse_args([str(ws), "--provider", "mock", *flags]), None)


# -- AC-4: web stays off unless --allow-web --------------------------------------


def test_web_absent_without_allow_web_even_if_configured(meeting_ws, fake_registered):
    configure(meeting_ws, {"search_provider": "fake"})
    agent = cli_agent(meeting_ws)
    assert "web" not in agent.tools
    assert agent.permissions.policy().get("web", Decision.DENY) is Decision.DENY


def test_allow_web_asks_and_bypass_allows(meeting_ws, fake_registered):
    configure(meeting_ws, {"search_provider": "fake"})
    asking = cli_agent(meeting_ws, "--allow-web")
    assert isinstance(asking.tools["web"], WebSearchTool) and asking.tools["web"].provider.info.name == "fake"
    assert asking.permissions.policy()["web"] is Decision.ASK
    bypass = cli_agent(meeting_ws, "--allow-web", "--permission-mode", "bypass")
    assert bypass.permissions.policy()["web"] is Decision.ALLOW


def test_the_sdk_default_permission_for_web_is_deny(meeting_ws, fake_registered):
    configure(meeting_ws, {"search_provider": "fake"})
    agent = Agent(meeting_ws, provider=MockProvider(), tools=["web"])
    assert agent.permissions.policy().get("web", Decision.DENY) is Decision.DENY


# -- AC-8: keys are never config values -------------------------------------------


def test_api_key_value_in_project_config_is_rejected(meeting_ws):
    configure(meeting_ws, {"search_provider": "brave", "search_providers": {"brave": {"api_key": "abc123"}}})
    with pytest.raises(ConfigurationError) as info:
        load_config(meeting_ws, user_config=None)
    message = str(info.value)
    assert "search_providers.brave.api_key must not be written" in message and "BRAVE_API_KEY" in message
    assert "abc123" not in message


def test_api_key_value_in_user_config_is_rejected(tmp_path):
    user = tmp_path / "settings.json"
    user.write_text(json.dumps({"search_providers": {"brave": {"api_key": "abc123"}}}))
    with pytest.raises(ConfigurationError, match="must not be written"):
        load_config(None, user_config=user)


def test_the_value_is_refused_when_building_too():
    """A config built in code (not read from a file) meets the same check."""
    with pytest.raises(ConfigurationError, match="must not be written"):
        registry.create("brave", {"api_key": "abc123"})


def test_secrets_override_reads_the_named_variable(meeting_ws, monkeypatch):
    monkeypatch.setenv("MY_BRAVE_KEY", "from-my-variable")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    configure(meeting_ws, {"search_provider": "brave", "search_providers": {"brave": {"secrets": {"api_key": "MY_BRAVE_KEY"}}}})
    agent = Agent(meeting_ws, provider=MockProvider(), tools=["web"])
    provider = agent.tools["web"].provider
    assert isinstance(provider, BraveProvider) and provider._api_key == "from-my-variable"


@pytest.mark.parametrize(
    "secrets,match",
    [
        ({"api_key": "not a variable name!"}, "must be an environment variable name"),
        ({"token": "X"}, "not a declared secret"),
        ("MY_KEY", "must be an object"),
    ],
)
def test_bad_secret_overrides_are_rejected(secrets, match):
    with pytest.raises(ConfigurationError, match=match):
        apply_config(KennelConfig(), {"search_providers": {"brave": {"secrets": secrets}}}, "<test>")


@pytest.mark.parametrize(
    "data,match",
    [
        ({"search_provider": 3}, "'search_provider' must be"),
        ({"search_providers": []}, "'search_providers' must be an object"),
        ({"search_providers": {"searxng": "http://x"}}, "search_providers.searxng must be an object"),
    ],
)
def test_malformed_search_config_is_rejected(data, match):
    with pytest.raises(ConfigurationError, match=match):
        apply_config(KennelConfig(), data, "<test>")


def test_project_options_merge_over_user_options(tmp_path):
    user = tmp_path / "settings.json"
    user.write_text(json.dumps({"search_provider": "searxng", "search_providers": {"searxng": {"url": "http://a", "timeout": 3}}}))
    ws = tmp_path / "ws"
    ws.mkdir()
    configure(ws, {"search_providers": {"searxng": {"url": "http://b"}}})
    cfg = load_config(ws, user_config=user)
    assert cfg.search_provider == "searxng"
    assert cfg.search_providers["searxng"] == {"url": "http://b", "timeout": 3}


def test_an_unknown_search_provider_is_a_configuration_error(meeting_ws):
    configure(meeting_ws, {"search_provider": "nope"})
    with pytest.raises(ConfigurationError, match="Unknown search provider 'nope'"):
        Agent(meeting_ws, provider=MockProvider(), tools=["web"])
    Agent(meeting_ws, provider=MockProvider())  # web not enabled: nothing is built


def test_no_provider_is_the_default():
    assert KennelConfig().search_provider is None
    assert set(registry.names()) >= {"searxng", "brave"}


# -- AC-13: public, pluggable, config-driven in the SDK ---------------------------------


def test_public_imports():
    for name in ("SearchProvider", "SearchResult", "SearchProviderInfo", "SearchError", "SearchSpec", "Secret", "register_search_provider"):
        assert name in kennel.__all__ and hasattr(kennel, name)
    assert issubclass(SearchError, ToolExecutionError)
    assert isinstance(Named(), SearchProvider)
    assert Secret("X").required


async def test_agent_dot_uses_the_kennel_json_search_provider(meeting_ws, fake_registered, monkeypatch):
    configure(meeting_ws, {"search_provider": "fake"})
    monkeypatch.chdir(meeting_ws)
    agent = Agent(".", provider=MockProvider(), tools=["web"], permissions={"web": "allow"})
    result = await agent.tools["web"].execute({"query": "q"}, agent.tool_context())
    assert result.metadata == {"provider": "fake", "returned": 1}


async def test_an_explicit_web_tool_instance_wins(meeting_ws, fake_registered):
    configure(meeting_ws, {"search_provider": "fake"})
    mine = WebSearchTool(Named("mine"))
    agent = Agent(meeting_ws, provider=MockProvider(), tools=["read", mine])
    assert agent.tools["web"] is mine
    result = await agent.tools["web"].execute({"query": "q"}, agent.tool_context())
    assert result.metadata["provider"] == "mine"


def test_readme_documents_a_custom_search_provider():
    readme = README.read_text(encoding="utf-8")
    assert "register_search_provider(SearchSpec(" in readme
    assert "class " in readme and "SearchProviderInfo(" in readme
    assert "raise SearchError(" in readme
