"""Web search for the ``web`` tool: the provider interface, the registry and the built-ins.

Kennel chooses no search provider by default (principle 1). ``searxng`` (a URL, no key)
and ``brave`` (one key, from the environment) are built in; anything else implements
:class:`SearchProvider` and is registered with :func:`register`.
"""

from ..credentials import Secret
from ..errors import SearchError
from .base import SearchProvider, SearchProviderInfo, SearchResult
from .registry import SearchSpec, create, names, register, spec

__all__ = [
    "SearchError",
    "SearchProvider",
    "SearchProviderInfo",
    "SearchResult",
    "SearchSpec",
    "Secret",
    "create",
    "names",
    "register",
    "spec",
]
