"""Model providers. :class:`AppleProvider` and ``LlamaServerProvider`` talk to real models;
:class:`MockProvider` is for tests.

Providers are chosen by name through :mod:`kennel.providers.registry`, which is also where
a new one is registered.
"""

from .base import ModelProvider, ProviderInfo, ProviderSession, ToolInvoker
from .mock import MockProvider
from .registry import DEFAULT_PROVIDER, ProviderSpec, create, names, register, spec

__all__ = [
    "DEFAULT_PROVIDER",
    "MockProvider",
    "ModelProvider",
    "ProviderInfo",
    "ProviderSession",
    "ProviderSpec",
    "ToolInvoker",
    "create",
    "names",
    "register",
    "spec",
]
