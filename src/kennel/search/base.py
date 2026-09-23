"""What a search provider is: the interface the ``web`` tool calls.

A provider is anything with an ``info`` and an async ``search``; it does not subclass
anything. Its promises:

* ``search`` returns an empty list only when it looked and found nothing;
* when it could not look (unreachable, rejected key, quota, a blocked upstream), it raises
  :class:`~kennel.errors.SearchError`, whose message says whether retrying can help;
* ``info.mode`` is ``"remote"`` unless the query itself never leaves the machine (an
  offline index). A SearXNG on localhost is still ``remote``: it relays the query upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchProviderInfo:
    """Who answers a search and where the query goes, as shown in the CLI header."""

    name: str
    destination: str
    mode: str = "remote"  # "local" only when the query never leaves this machine

    def describe(self) -> str:
        return f"{self.name} → {self.destination} ({self.mode})"


@runtime_checkable
class SearchProvider(Protocol):
    info: SearchProviderInfo

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...
