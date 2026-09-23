"""Secrets come from the environment and never show up as values (issue #61, AC-3/AC-8)."""

import pytest

from kennel import ConfigurationError, Secret
from kennel.credentials import (
    MissingSecretError,
    check_no_secret_values,
    resolve_secrets,
    secret_status,
)

DECLARED = {"api_key": Secret("SVC_KEY"), "extra": Secret("SVC_EXTRA", required=False)}
WHERE = "search_providers.svc"


def test_values_are_read_by_argument_name():
    env = {"SVC_KEY": "k", "SVC_EXTRA": "e"}
    assert resolve_secrets(DECLARED, where=WHERE, environ=env) == {"api_key": "k", "extra": "e"}


def test_an_unset_optional_secret_is_left_out():
    assert resolve_secrets(DECLARED, where=WHERE, environ={"SVC_KEY": "k"}) == {"api_key": "k"}


def test_a_missing_required_secret_names_the_variable():
    with pytest.raises(MissingSecretError) as info:
        resolve_secrets(DECLARED, where=WHERE, environ={})
    message = str(info.value)
    assert "SVC_KEY is not set" in message and f"{WHERE}.secrets.api_key" in message
    assert isinstance(info.value, ConfigurationError)


def test_an_override_names_another_variable():
    env = {"MINE": "m"}
    assert resolve_secrets(DECLARED, {"api_key": "MINE"}, where=WHERE, environ=env) == {"api_key": "m"}
    with pytest.raises(MissingSecretError, match="MINE is not set"):
        resolve_secrets(DECLARED, {"api_key": "MINE"}, where=WHERE, environ={"SVC_KEY": "k"})


def test_a_value_where_a_variable_name_belongs_is_refused_without_echoing_it():
    with pytest.raises(ConfigurationError) as info:
        resolve_secrets(DECLARED, {"api_key": "sk-live 123"}, where=WHERE, environ={})
    assert "sk-live 123" not in str(info.value)


def test_a_declared_secret_as_an_option_is_refused():
    with pytest.raises(ConfigurationError) as info:
        check_no_secret_values(DECLARED, {"api_key": "sk-live-123", "url": "x"}, where=WHERE)
    message = str(info.value)
    assert f"{WHERE}.api_key must not be written" in message and "SVC_KEY" in message
    assert "sk-live-123" not in message
    check_no_secret_values(DECLARED, {"url": "x"}, where=WHERE)  # other options are fine


def test_status_reports_presence_only():
    statuses = secret_status(DECLARED, where=WHERE, environ={"SVC_KEY": "sk-live-123"})
    assert [(s.param, s.env, s.present, s.required) for s in statuses] == [
        ("api_key", "SVC_KEY", True, True),
        ("extra", "SVC_EXTRA", False, False),
    ]
    assert "sk-live-123" not in repr(statuses)
