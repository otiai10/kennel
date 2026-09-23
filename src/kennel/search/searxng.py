"""SearXNG: a self-hosted metasearch engine, reached at a URL the user gives.

``GET {url}/search?q=...&format=json``. SearXNG relays each query to upstream engines, so
it is ``remote`` even on localhost. Things that go wrong in practice, each mapped to a
:class:`~kennel.errors.SearchError` that says what to do:

* JSON output is off by default and a disabled format answers HTTP 403
  (https://docs.searxng.org/dev/search_api.html);
* its limiter answers HTTP 429;
* upstream engines block it (CAPTCHA, access denied, timeouts). The response then lists
  them in ``unresponsive_engines`` as ``[engine, reason]`` pairs -- that shape is read
  from SearXNG's own JSON output, not from its documentation, so anything else there is
  ignored. No results *and* unresponsive engines is a failure, not "nothing found".
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..diagnostics import Check
from ..errors import ConfigurationError, SearchError
from .base import SearchProviderInfo, SearchResult
from .http import DEFAULT_MAX_BYTES, DEFAULT_TIMEOUT, Endpoint, http_get

SERVICE = "SearXNG"
JSON_DISABLED_REMEDY = "Add json to search.formats in the SearXNG settings.yml and restart it."
_URL_REMEDY = "Check that search_providers.searxng.url points at a SearXNG instance."


class SearxngProvider:
    def __init__(
        self,
        url: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if not url or not isinstance(url, str):
            raise ConfigurationError("search_providers.searxng.url is required (for example http://localhost:8888)")
        self.endpoint = Endpoint.parse(url, what="search_providers.searxng.url")
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)
        self.info = SearchProviderInfo("searxng", f"{self.endpoint.netloc} → upstream engines", "remote")

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        response = await http_get(
            self.endpoint,
            "/search",
            {"q": query, "format": "json"},
            timeout=self.timeout,
            max_bytes=self.max_response_bytes,
            service=SERVICE,
        )
        if response.status == 403:
            raise SearchError(
                "SearXNG refused JSON output (HTTP 403); its json format is probably disabled",
                retryable=False,
                remedy=JSON_DISABLED_REMEDY,
            )
        if response.status == 429:
            raise SearchError(
                "SearXNG's limiter rejected the request (HTTP 429)",
                retryable=False,
                remedy="Turn server.limiter off in settings.yml for a private instance.",
            )
        if response.status >= 500:
            raise SearchError(f"SearXNG answered HTTP {response.status}", retryable=True)
        if response.status != 200:
            raise SearchError(f"SearXNG answered HTTP {response.status}", retryable=False, remedy=_URL_REMEDY)
        try:
            data = json.loads(response.body.decode("utf-8", "replace"))
        except ValueError:
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SearchError("SearXNG sent a response that is not its JSON format", retryable=False, remedy=_URL_REMEDY)
        results = [_result(item) for item in data["results"] if isinstance(item, dict)]
        results = [r for r in results if r.url][:limit]
        if not results:
            failed = _unresponsive(data.get("unresponsive_engines"))
            if failed:
                raise SearchError(
                    f"no SearXNG engine answered ({failed})",
                    retryable=False,
                    remedy="Upstream engines may be blocking this instance; check its engines in settings.yml.",
                )
        return results


def _result(item: dict[str, Any]) -> SearchResult:
    return SearchResult(
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        snippet=str(item.get("content") or ""),
    )


def _unresponsive(raw: Any) -> str:
    """``"google: CAPTCHA, bing: timeout"`` from ``[[engine, reason], ...]``; ``""`` otherwise."""
    if not isinstance(raw, list):
        return ""
    pairs = [f"{item[0]}: {item[1]}" for item in raw if isinstance(item, (list, tuple)) and len(item) >= 2]
    return ", ".join(pairs)


def searxng_doctor_checks(provider: SearxngProvider) -> list[Check]:
    """Send one probe query. It goes through the instance to its upstream engines, and says so."""
    where = f"SearXNG at {provider.endpoint.netloc}"
    try:
        asyncio.run(provider.search("kennel", limit=1))
    except SearchError as exc:
        hint = exc.remedy or "check that the SearXNG instance is running"
        return [Check("web search reachable", False, f"{where}: {exc} (tried a probe query)", hint=hint)]
    return [Check("web search reachable", True, f"{where} answered JSON (probe query sent via upstream engines)")]
