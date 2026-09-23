"""Model providers declare secrets the way search providers do (issue #74).

The key is read from an environment variable only; a declared secret written into the config
is refused both when the file is read and when a provider is built from a config made in code.
"""

import json
from pathlib import Path

import pytest

from kennel import Agent, ConfigurationError, KennelConfig, ProviderSpec, Secret
from kennel.config import load_config
from kennel.providers.mock import MockProvider
from kennel.providers.registry import _SPECS, create, register, resolve_options, spec

KEY = "sk-must-not-be-echoed"


@pytest.fixture
def keyed_provider():
    """A third-party provider with one required and one optional secret."""
    built: list[dict] = []

    def factory(**options):
        built.append(options)
        return MockProvider(["ok"])

    register(
        ProviderSpec(
            "keyed",
            factory,
            secrets={"token": Secret("KEYED_TOKEN"), "extra": Secret("KEYED_EXTRA", required=False)},
        )
    )
    try:
        yield built
    finally:
        _SPECS.pop("keyed", None)


# -- AC-4: a key written as a value is refused, from a file or from code ----------------


def test_a_key_in_kennel_json_is_refused_on_load(meeting_ws: Path):
    (meeting_ws / "kennel.json").write_text(json.dumps({"providers": {"llama-server": {"api_key": KEY}}}))
    with pytest.raises(ConfigurationError) as info:
        load_config(meeting_ws, user_config=None)
    message = str(info.value)
    assert "providers.llama-server.api_key" in message and "KENNEL_LLAMA_SERVER_API_KEY" in message
    assert "environment" in message and KEY not in message


def test_a_key_in_a_config_built_in_code_is_refused_when_the_provider_is_built(meeting_ws: Path):
    config = KennelConfig(provider="llama-server", providers={"llama-server": {"api_key": KEY}})
    with pytest.raises(ConfigurationError) as info:
        Agent(meeting_ws, config=config)
    assert "providers.llama-server.api_key" in str(info.value) and KEY not in str(info.value)
    with pytest.raises(ConfigurationError):
        create("llama-server", api_key=KEY)


@pytest.mark.parametrize(
    ("secrets", "fragment"),
    [
        ({"token": "KEYED_TOKEN"}, "not a declared secret"),
        ({"api_key": "sk-a value, not a name"}, "must be an environment variable name"),
        ("LLAMA_API_KEY", "must be an object of variable names"),
    ],
)
def test_bad_secrets_entries_are_refused_on_load(meeting_ws: Path, secrets, fragment: str):
    (meeting_ws / "kennel.json").write_text(json.dumps({"providers": {"llama-server": {"secrets": secrets}}}))
    with pytest.raises(ConfigurationError, match=fragment):
        load_config(meeting_ws, user_config=None)


def test_secrets_reach_the_factory_under_their_argument_names(keyed_provider, monkeypatch):
    monkeypatch.setenv("KEYED_TOKEN", "t0ken")
    monkeypatch.delenv("KEYED_EXTRA", raising=False)
    create("keyed", flavour="plain")
    assert keyed_provider == [{"flavour": "plain", "token": "t0ken"}]  # an unset optional one is left out


def test_a_missing_required_secret_names_the_variable(keyed_provider, monkeypatch):
    monkeypatch.delenv("KEYED_TOKEN", raising=False)
    with pytest.raises(ConfigurationError, match="KEYED_TOKEN is not set"):
        create("keyed")


def test_resolve_options_is_what_doctor_and_create_share(keyed_provider, monkeypatch):
    monkeypatch.setenv("OTHER_TOKEN", "t0ken")
    options = {"secrets": {"token": "OTHER_TOKEN"}, "flavour": "plain"}
    assert resolve_options("keyed", options) == {"flavour": "plain", "token": "t0ken"}
    assert options == {"secrets": {"token": "OTHER_TOKEN"}, "flavour": "plain"}  # not mutated


# -- AC-6: the positional form still works; the built-in providers declare nothing ----


def test_the_positional_provider_spec_keeps_its_meaning():
    checks = lambda **_: []  # noqa: E731
    positional = ProviderSpec("p", MockProvider, checks)
    assert positional.doctor_checks is checks and positional.secrets == {}


@pytest.mark.parametrize("name", ["apple", "llama-cpp", "mock"])
def test_only_llama_server_declares_a_secret(name: str):
    assert spec(name).secrets == {}


def test_llama_server_declares_an_optional_key():
    from kennel.providers.llama_server import API_KEY_ENV

    assert dict(spec("llama-server").secrets) == {"api_key": Secret("KENNEL_LLAMA_SERVER_API_KEY", required=False)}
    assert API_KEY_ENV == "KENNEL_LLAMA_SERVER_API_KEY"  # the 401 message names the same default


def test_options_of_a_provider_without_secrets_pass_through_unchanged():
    script = json.dumps({"turns": ["scripted"]})
    assert resolve_options("mock", {"script": script}) == {"script": script}
    assert resolve_options("mock", {"script": script, "secrets": {}}) == {"script": script}
