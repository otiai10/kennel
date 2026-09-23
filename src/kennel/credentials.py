"""Secrets (API keys) are read from environment variables, never from a config file.

A registry entry declares which factory arguments are secrets and which environment
variable holds each one by default::

    secrets={"api_key": Secret("BRAVE_API_KEY")}

Kennel reads the variable and passes the value to the factory under the argument's name,
so the code that uses the key never learns where it came from. The config file may only
*name* a different variable (``{"secrets": {"api_key": "MY_BRAVE_KEY"}}``); a declared
secret written there as a value is refused. The workspace's ``kennel.json`` is readable by
the agent's own ``read`` tool, so a key kept there could be read out by a prompt injection
and sent off in a query.

Nothing here ever puts a secret's *value* into a message: errors and diagnostics name the
variable only. The module knows nothing about search or models: the search registry and the
model provider registry both go through :func:`check_options` and :func:`with_secrets`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ConfigurationError

__all__ = [
    "SECRETS_KEY",
    "MissingSecretError",
    "Secret",
    "SecretStatus",
    "check_no_secret_values",
    "check_options",
    "resolve_secrets",
    "secret_status",
    "variable_names",
    "with_secrets",
]

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: The config key under which a variable name may replace a secret's default variable.
SECRETS_KEY = "secrets"


class MissingSecretError(ConfigurationError):
    """A required secret's environment variable is not set. The message names the variable."""


@dataclass(frozen=True)
class Secret:
    """A factory argument whose value comes from the environment variable ``env``."""

    env: str
    required: bool = True


@dataclass(frozen=True)
class SecretStatus:
    """Whether a secret's variable is set. Carries the variable's name, never its value."""

    param: str
    env: str
    present: bool
    required: bool


def _how_to_set(where: str, param: str, env: str) -> str:
    return f"set {env} in the environment, or name another variable in {where}.{SECRETS_KEY}.{param}"


def check_no_secret_values(declared: Mapping[str, Secret], options: Mapping[str, Any], *, where: str) -> None:
    """Refuse options that write a declared secret as a value (instead of naming a variable)."""
    for param, secret in declared.items():
        if param in options:
            raise ConfigurationError(
                f"{where}.{param} must not be written in a config file (the agent can read "
                f"kennel.json); {_how_to_set(where, param, secret.env)}"
            )


def variable_names(
    declared: Mapping[str, Secret], overrides: Any, *, where: str
) -> dict[str, str]:
    """The environment variable each declared secret is read from, overrides applied."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ConfigurationError(f"{where}.{SECRETS_KEY} must be an object of variable names")
    for param, env in overrides.items():
        if param not in declared:
            known = ", ".join(declared) or "none"
            raise ConfigurationError(f"{where}.{SECRETS_KEY}.{param}: not a declared secret (declared: {known})")
        if not isinstance(env, str) or not _ENV_NAME.match(env):
            raise ConfigurationError(
                f"{where}.{SECRETS_KEY}.{param} must be an environment variable name, not a value"
            )
    return {param: overrides.get(param, secret.env) for param, secret in declared.items()}


def resolve_secrets(
    declared: Mapping[str, Secret],
    overrides: Any = None,
    *,
    where: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read every declared secret from the environment, keyed by factory argument name.

    A missing required secret raises :class:`ConfigurationError` naming the variable; an
    unset optional one is left out, so the factory's own default applies.
    """
    env = os.environ if environ is None else environ
    values: dict[str, str] = {}
    for param, name in variable_names(declared, overrides, where=where).items():
        value = env.get(name, "")
        if value:
            values[param] = value
        elif declared[param].required:
            raise MissingSecretError(f"{name} is not set; {_how_to_set(where, param, name)}")
    return values


def secret_status(
    declared: Mapping[str, Secret],
    overrides: Any = None,
    *,
    where: str,
    environ: Mapping[str, str] | None = None,
) -> list[SecretStatus]:
    """Which declared secrets are set, for diagnostics. Values are never returned."""
    env = os.environ if environ is None else environ
    return [
        SecretStatus(param, name, bool(env.get(name)), declared[param].required)
        for param, name in variable_names(declared, overrides, where=where).items()
    ]


def check_options(declared: Mapping[str, Secret], options: Mapping[str, Any], *, where: str) -> None:
    """Refuse a declared secret written as a value, or a ``secrets`` entry that is not a
    declared secret's variable name. Reads no environment variable."""
    check_no_secret_values(declared, options, where=where)
    variable_names(declared, options.get(SECRETS_KEY), where=where)


def with_secrets(
    declared: Mapping[str, Secret],
    options: Mapping[str, Any] | None,
    *,
    where: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The factory's keyword arguments: ``options`` without ``secrets``, plus every declared
    secret read from the environment (see :func:`resolve_secrets`)."""
    opts = dict(options or {})
    overrides = opts.pop(SECRETS_KEY, None)
    check_no_secret_values(declared, opts, where=where)
    return {**opts, **resolve_secrets(declared, overrides, where=where, environ=environ)}
