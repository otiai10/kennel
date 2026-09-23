"""The one place search providers are registered and built by name.

The shape follows :mod:`kennel.providers.registry`: a static dict plus :func:`register`.
The config selects one with ``search_provider`` (a name) and gives options per name under
``search_providers``::

    {
      "search_provider": "brave",
      "search_providers": {"brave": {"secrets": {"api_key": "MY_BRAVE_KEY"}}}
    }

A :class:`SearchSpec` declares which factory arguments are secrets. :func:`create` reads
those from the environment (:mod:`kennel.credentials`) and refuses them as config values,
so a provider's own code never sees where its key came from. No provider is the default:
without ``search_provider`` the ``web`` tool answers "not configured".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..credentials import (
    SECRETS_KEY,
    Secret,
    check_no_secret_values,
    resolve_secrets,
    variable_names,
)
from ..diagnostics import Check
from ..errors import ConfigurationError
from .base import SearchProvider

SearchFactory = Callable[..., SearchProvider]
SearchDoctorChecks = Callable[[SearchProvider], Sequence[Check]]


@dataclass(frozen=True)
class SearchSpec:
    """A search provider that can be chosen by name.

    ``factory(**options, **secrets)`` builds it: ``options`` come from
    ``search_providers.<name>`` in the config, ``secrets`` from the environment variables
    declared here. ``doctor_checks(provider)`` optionally tells ``kennel doctor`` whether the
    built provider can be reached.
    """

    name: str
    factory: SearchFactory
    secrets: Mapping[str, Secret] = field(default_factory=dict)
    doctor_checks: SearchDoctorChecks | None = None


def _searxng_factory(**options: Any) -> SearchProvider:
    from .searxng import SearxngProvider

    return SearxngProvider(**options)


def _searxng_doctor_checks(provider: SearchProvider) -> Sequence[Check]:
    from .searxng import searxng_doctor_checks

    return searxng_doctor_checks(provider)


def _brave_factory(**options: Any) -> SearchProvider:
    from .brave import BraveProvider

    return BraveProvider(**options)


def _brave_doctor_checks(provider: SearchProvider) -> Sequence[Check]:
    from .brave import brave_doctor_checks

    return brave_doctor_checks(provider)


_SPECS: dict[str, SearchSpec] = {
    "searxng": SearchSpec("searxng", _searxng_factory, doctor_checks=_searxng_doctor_checks),
    "brave": SearchSpec(
        "brave", _brave_factory, secrets={"api_key": Secret("BRAVE_API_KEY")}, doctor_checks=_brave_doctor_checks
    ),
}


def register(spec: SearchSpec) -> None:
    """Add (or replace) a search provider so it can be chosen by name."""
    _SPECS[spec.name] = spec


def names() -> tuple[str, ...]:
    return tuple(_SPECS)


def spec(name: str) -> SearchSpec:
    try:
        return _SPECS[name]
    except KeyError:
        raise ConfigurationError(
            f"Unknown search provider {name!r}. Registered search providers: {', '.join(names())}"
        ) from None


def where(name: str) -> str:
    """The config path of a search provider's options, as named in messages."""
    return f"search_providers.{name}"


def check_options(name: str, options: Mapping[str, Any]) -> None:
    """Refuse a declared secret written as a value, or a ``secrets`` entry that is not a
    variable name. A no-op for names not registered yet (:func:`create` checks them)."""
    if name in _SPECS:
        declared = _SPECS[name].secrets
        check_no_secret_values(declared, options, where=where(name))
        variable_names(declared, options.get(SECRETS_KEY), where=where(name))


def create(name: str, options: Mapping[str, Any] | None = None) -> SearchProvider:
    """Build the search provider ``name`` from its config options and its secrets.

    A required secret whose variable is unset raises
    :class:`~kennel.credentials.MissingSecretError` (a :class:`ConfigurationError`) naming
    the variable, never a value.
    """
    search_spec = spec(name)
    opts = dict(options or {})
    overrides = opts.pop(SECRETS_KEY, None)
    check_no_secret_values(search_spec.secrets, opts, where=where(name))
    secrets = resolve_secrets(search_spec.secrets, overrides, where=where(name))
    try:
        return search_spec.factory(**opts, **secrets)
    except TypeError as exc:
        raise ConfigurationError(f"search provider {name!r}: {exc}") from exc
