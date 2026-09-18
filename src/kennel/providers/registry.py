"""The one place providers are registered, so ``Agent``, the CLI and ``kennel doctor`` agree.

A provider is named by a string everywhere it can be chosen (``--provider``, the
``provider`` config key, ``Agent(provider=)``), and that string is resolved here::

    from kennel.providers.registry import ProviderSpec, register, create

    register(ProviderSpec("echo", lambda **options: EchoProvider(**options)))
    provider = create("echo")

``ProviderSpec.factory`` takes provider-specific keyword arguments, which come from
``providers.<name>`` in the config file. ``doctor_checks`` is optional and returns the
:class:`~kennel.diagnostics.Check` list ``kennel doctor`` shows for that provider.

Registration is a static dict plus :func:`register`; entry points are deliberately not
read (a provider has to be imported to be usable anyway, and tests stay independent of
what happens to be installed).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..diagnostics import Check
from ..errors import ConfigurationError
from .base import ModelProvider

#: The provider used when nothing selects one. Apple's on-device model is Kennel's reason to exist.
DEFAULT_PROVIDER = "apple"

ProviderFactory = Callable[..., ModelProvider]
DoctorChecks = Callable[..., Sequence[Check]]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    factory: ProviderFactory
    doctor_checks: DoctorChecks | None = field(default=None)


def _apple_factory(**options: Any) -> ModelProvider:
    from .apple import apple_provider

    return apple_provider(**options)


def _apple_doctor_checks(**options: Any) -> Sequence[Check]:
    from .apple import apple_doctor_checks

    return apple_doctor_checks(**options)


def _mock_factory(script: str | dict[str, Any] | None = None) -> ModelProvider:
    from .mock import MockProvider

    return MockProvider() if script is None else MockProvider.from_json(script)


_SPECS: dict[str, ProviderSpec] = {
    "apple": ProviderSpec("apple", _apple_factory, _apple_doctor_checks),
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


def create(name: str, **options: Any) -> ModelProvider:
    """Build the provider called ``name`` with its configured options."""
    factory = spec(name).factory
    try:
        return factory(**options)
    except TypeError as exc:
        raise ConfigurationError(f"provider {name!r}: {exc}") from exc
