"""The one place providers are registered, so ``Agent``, the CLI and ``kennel doctor`` agree.

A provider is named by a string everywhere it can be chosen (``--provider``, the
``provider`` config key, ``Agent(provider=)``), and that string is resolved here::

    from kennel.providers.registry import ProviderSpec, register, create

    register(ProviderSpec("echo", lambda **options: EchoProvider(**options)))
    provider = create("echo")

``ProviderSpec.factory`` takes provider-specific keyword arguments, which come from
``providers.<name>`` in the config file. ``doctor_checks`` is optional and returns the
:class:`~kennel.diagnostics.Check` list ``kennel doctor`` shows for that provider.
``secrets`` declares which factory arguments are secrets (API keys): they are read from
environment variables and refused as config values (:mod:`kennel.credentials`), the same
way search providers declare theirs. :func:`resolve_options` is the one place that turns
config options into factory arguments, for :func:`create` and ``kennel doctor`` alike.

Registration is a static dict plus :func:`register`; entry points are deliberately not
read (a provider has to be imported to be usable anyway, and tests stay independent of
what happens to be installed).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import credentials
from ..credentials import Secret
from ..diagnostics import Check
from ..errors import ConfigurationError
from .base import ModelProvider

#: The provider used when nothing selects one. Apple's on-device model is Kennel's reason to exist.
DEFAULT_PROVIDER = "apple"

ProviderFactory = Callable[..., ModelProvider]
DoctorChecks = Callable[..., Sequence[Check]]


@dataclass(frozen=True)
class ProviderSpec:
    """A model provider that can be chosen by name.

    ``factory(**options)`` builds it and ``doctor_checks(**options)`` optionally checks it,
    where ``options`` are ``providers.<name>`` from the config plus the ``secrets`` declared
    here, read from the environment. ``secrets`` is last so that the positional
    ``ProviderSpec(name, factory, doctor_checks)`` keeps its meaning.
    """

    name: str
    factory: ProviderFactory
    doctor_checks: DoctorChecks | None = field(default=None)
    secrets: Mapping[str, Secret] = field(default_factory=dict)


def _apple_factory(**options: Any) -> ModelProvider:
    from .apple import apple_provider

    return apple_provider(**options)


def _apple_doctor_checks(**options: Any) -> Sequence[Check]:
    from .apple import apple_doctor_checks

    return apple_doctor_checks(**options)


def _llama_server_factory(**options: Any) -> ModelProvider:
    from .llama_server import llama_server_provider

    return llama_server_provider(**options)


def _llama_server_doctor_checks(**options: Any) -> Sequence[Check]:
    from .llama_server import llama_server_doctor_checks

    return llama_server_doctor_checks(**options)


def _llama_cpp_factory(**options: Any) -> ModelProvider:
    from .llama_cpp import llama_cpp_provider

    return llama_cpp_provider(**options)


def _llama_cpp_doctor_checks(**options: Any) -> Sequence[Check]:
    from .llama_cpp import llama_cpp_doctor_checks

    return llama_cpp_doctor_checks(**options)


def _mock_factory(script: str | dict[str, Any] | None = None) -> ModelProvider:
    from .mock import MockProvider

    return MockProvider() if script is None else MockProvider.from_json(script)


_SPECS: dict[str, ProviderSpec] = {
    "apple": ProviderSpec("apple", _apple_factory, _apple_doctor_checks),
    "llama-server": ProviderSpec(
        "llama-server",
        _llama_server_factory,
        _llama_server_doctor_checks,
        secrets={"api_key": Secret("KENNEL_LLAMA_SERVER_API_KEY", required=False)},
    ),
    "llama-cpp": ProviderSpec("llama-cpp", _llama_cpp_factory, _llama_cpp_doctor_checks),
    "mock": ProviderSpec("mock", _mock_factory),
}


def register(spec: ProviderSpec) -> None:
    """Add (or replace) a provider so it can be chosen by name."""
    _SPECS[spec.name] = spec


def names() -> tuple[str, ...]:
    """Every registered provider name, in registration order."""
    return tuple(_SPECS)


def spec(name: str) -> ProviderSpec:
    try:
        return _SPECS[name]
    except KeyError:
        raise ConfigurationError(
            f"Unknown provider {name!r}. Registered providers: {', '.join(names())}"
        ) from None


def where(name: str) -> str:
    """The config path of a provider's options, as named in messages."""
    return f"providers.{name}"


def check_options(name: str, options: Mapping[str, Any]) -> None:
    """Refuse a declared secret written as a value, or a ``secrets`` entry that is not a
    variable name. A no-op for names not registered yet (:func:`create` checks them)."""
    if name in _SPECS:
        credentials.check_options(_SPECS[name].secrets, options, where=where(name))


def resolve_options(name: str, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The keyword arguments for ``name``'s factory and doctor checks: the config options
    without ``secrets``, plus each declared secret read from its environment variable.

    Raises :class:`ConfigurationError` for a secret written as a value (also when the
    config was built in code and never read from a file) or a required one that is unset.
    """
    return credentials.with_secrets(spec(name).secrets, options, where=where(name))


def create(name: str, **options: Any) -> ModelProvider:
    """Build the provider called ``name`` with its configured options and its secrets."""
    factory = spec(name).factory
    kwargs = resolve_options(name, options)
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise ConfigurationError(f"provider {name!r}: {exc}") from exc
