"""Choosing a provider by name from the config file, the CLI and Agent() (issue #35)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from kennel import Agent, ConfigurationError, ModelUnavailableError, ProviderSpec
from kennel.cli.main import build_agent, build_parser
from kennel.providers.base import ModelProvider, ProviderInfo
from kennel.providers.mock import MockProvider
from kennel.providers.registry import _SPECS, create, names, register


class FakeProvider(MockProvider):
    def __init__(self, **options):
        super().__init__(["fake answer"])
        self.options = options
        self.info = ProviderInfo(name="fake", model="FakeModel")


@pytest.fixture
def fake_provider():
    """Register a third-party provider and take it back out again."""
    register(ProviderSpec("fake", lambda **options: FakeProvider(**options), lambda **_: []))
    try:
        yield "fake"
    finally:
        _SPECS.pop("fake", None)


def _write_config(ws: Path, data: dict) -> None:
    (ws / "kennel.json").write_text(json.dumps(data))


# -- AC-1 / AC-2: the config file selects the provider and carries its options ----------


def test_project_config_selects_the_provider(meeting_ws: Path):
    _write_config(meeting_ws, {"provider": "mock"})
    assert Agent(meeting_ws).provider.info.name == "mock"


def test_provider_options_reach_the_factory(meeting_ws: Path):
    script = json.dumps({"turns": ["scripted answer"]})
    _write_config(meeting_ws, {"provider": "mock", "providers": {"mock": {"script": script}}})
    agent = Agent(meeting_ws)
    assert asyncio.run(agent.run("anything")).text == "scripted answer"


def test_options_of_other_providers_are_ignored(meeting_ws: Path, fake_provider: str):
    """Only the chosen provider's options are used, so `--provider` can switch names alone."""
    _write_config(
        meeting_ws,
        {"provider": "fake", "providers": {"fake": {"flavour": "vanilla"}, "mock": {"script": "{"}}},
    )
    agent = Agent(meeting_ws)
    assert agent.provider.info.name == "fake" and agent.provider.options == {"flavour": "vanilla"}


# -- AC-3: precedence between an instance, a name and the CLI flag ----------------------


def test_an_instance_beats_the_config(meeting_ws: Path):
    _write_config(meeting_ws, {"provider": "mock"})
    agent = Agent(meeting_ws, provider=FakeProvider())
    assert isinstance(agent.provider, ModelProvider) and agent.provider.info.name == "fake"


def test_a_name_is_resolved_by_the_registry(meeting_ws: Path):
    assert Agent(meeting_ws, provider="mock").provider.info.name == "mock"


def test_cli_flag_beats_the_config(meeting_ws: Path, fake_provider: str):
    _write_config(meeting_ws, {"provider": "fake"})
    args = build_parser().parse_args([str(meeting_ws), "--provider", "mock"])
    assert build_agent(args, None).provider.info.name == "mock"


def test_without_the_flag_the_config_wins(meeting_ws: Path, fake_provider: str):
    _write_config(meeting_ws, {"provider": "fake"})
    args = build_parser().parse_args([str(meeting_ws)])
    assert args.provider is None  # so that the config, not the flag default, decides
    assert build_agent(args, None).provider.info.name == "fake"


def test_mock_script_from_the_environment_becomes_a_provider_option(meeting_ws: Path, monkeypatch):
    monkeypatch.setenv("KENNEL_MOCK_SCRIPT", json.dumps({"turns": ["from the environment"]}))
    args = build_parser().parse_args([str(meeting_ws), "--provider", "mock"])
    agent = build_agent(args, None)
    assert asyncio.run(agent.run("x")).text == "from the environment"


# -- AC-4: unknown names and malformed options are configuration errors -----------------


def test_unknown_name_in_the_config_lists_the_registered_ones(meeting_ws: Path):
    _write_config(meeting_ws, {"provider": "nope"})
    with pytest.raises(ConfigurationError) as exc:
        Agent(meeting_ws)
    assert "nope" in str(exc.value)
    assert all(name in str(exc.value) for name in names())


def test_unknown_name_as_an_argument(meeting_ws: Path):
    with pytest.raises(ConfigurationError, match="Unknown provider 'nope'"):
        Agent(meeting_ws, provider="nope")


@pytest.mark.parametrize(
    "data",
    [
        {"provider": 1},
        {"providers": []},
        {"providers": {"mock": "script"}},
    ],
)
def test_malformed_provider_config(meeting_ws: Path, data: dict):
    _write_config(meeting_ws, data)
    with pytest.raises(ConfigurationError):
        Agent(meeting_ws)


def test_unknown_option_key_is_a_configuration_error(meeting_ws: Path):
    _write_config(meeting_ws, {"provider": "mock", "providers": {"mock": {"nope": 1}}})
    with pytest.raises(ConfigurationError, match="provider 'mock'"):
        Agent(meeting_ws)


# -- AC-5: Kennel works where apple_fm_sdk cannot be imported --------------------------


def test_selection_works_without_apple_fm_sdk(meeting_ws: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "apple_fm_sdk", None)  # makes `import apple_fm_sdk` fail
    import kennel  # noqa: F401 - importing Kennel itself must not need the SDK

    assert Agent(meeting_ws, provider="mock").provider.info.name == "mock"
    with pytest.raises(ModelUnavailableError) as exc:
        Agent(meeting_ws, provider="apple")
    assert "--provider" in str(exc.value)


# -- AC-6: a registered provider is selectable everywhere -------------------------------


def test_registered_provider_is_selectable(meeting_ws: Path, fake_provider: str):
    assert Agent(meeting_ws, provider="fake").provider.info.name == "fake"
    assert "fake" in names() and isinstance(create("fake"), FakeProvider)
    assert "fake" in build_parser().parse_args([str(meeting_ws), "--provider", "fake"]).provider
