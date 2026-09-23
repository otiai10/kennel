"""Brave Search API: a hosted index, one API key (``BRAVE_API_KEY``).

``GET https://api.search.brave.com/res/v1/web/search?q=...&count=N`` with the key in the
``X-Subscription-Token`` header; results are ``web.results[]`` with ``title``, ``url`` and
``description`` (Brave's web search documentation). The key only ever travels in that
header, never in a URL or a message.

Error mapping. Brave documents HTTP 429 for exceeding a limit and the ``X-RateLimit-*``
headers, and says both the per-second rate and the monthly quota answer 429; the
``X-RateLimit-Remaining`` header (``"<per second>, <per month>"``) tells them apart. The rest
is **assumed** from reports of the live API, not from its documentation: 401/403 or 422 with
``error.code == "SUBSCRIPTION_TOKEN_INVALID"`` for a rejected key, ``error.code ==
"QUOTA_LIMITED"`` for a spent quota, and 402 for a plan that cannot pay. Anything that
matches none of them falls back to a plain "HTTP <status>" error.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from ..diagnostics import Check
from ..errors import SearchError
from .base import SearchProviderInfo, SearchResult
from .http import DEFAULT_MAX_BYTES, DEFAULT_TIMEOUT, Endpoint, http_get, probe_connection

SERVICE = "Brave Search"
BASE_URL = "https://api.search.brave.com"
_MAX_COUNT = 20
_TAG = re.compile(r"<[^>]+>")

_AUTH_REMEDY = "Check the Brave API key Kennel was given and that its plan is active."
_QUOTA_REMEDY = "The plan's quota is used up; raise the limit in the Brave Search API dashboard."


class BraveProvider:
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_BYTES,
        endpoint: str = BASE_URL,  # tests point this at a local fake server
    ) -> None:
        self._api_key = api_key
        self.endpoint = Endpoint.parse(endpoint, what="brave endpoint")
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)
        self.info = SearchProviderInfo("brave", self.endpoint.netloc, "remote")

    def __repr__(self) -> str:  # never show the key
        return f"BraveProvider(endpoint={self.endpoint.netloc!r})"

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        response = await http_get(
            self.endpoint,
            "/res/v1/web/search",
            {"q": query, "count": max(1, min(limit, _MAX_COUNT))},
            headers={"X-Subscription-Token": self._api_key},
            timeout=self.timeout,
            max_bytes=self.max_response_bytes,
            service=SERVICE,
        )
        data = _json(response.body)
        if response.status != 200:
            raise _http_error(response.status, response.headers, data)
        if not isinstance(data, dict):
            raise SearchError("Brave Search sent a response that is not JSON", retryable=True)
        web = data.get("web")
        items = web.get("results") if isinstance(web, dict) else None
        if not isinstance(items, list):
            return []  # Brave leaves "web" out when nothing matched
        results = [_result(item) for item in items if isinstance(item, dict)]
        return [r for r in results if r.url][:limit]


def _json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return None


def _text(value: Any) -> str:
    return html.unescape(_TAG.sub("", str(value or "")))


def _result(item: dict[str, Any]) -> SearchResult:
    return SearchResult(title=_text(item.get("title")), url=str(item.get("url") or ""), snippet=_text(item.get("description")))


def _monthly_quota_spent(headers: Any) -> bool:
    """``X-RateLimit-Remaining: "1, 0"`` -- the second window (the month) has nothing left."""
    remaining = [part.strip() for part in str(headers.get("x-ratelimit-remaining", "")).split(",")]
    return len(remaining) >= 2 and remaining[1] == "0"


def _http_error(status: int, headers: Any, data: Any) -> SearchError:
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if status in (401, 403) or code == "SUBSCRIPTION_TOKEN_INVALID":
        return SearchError(f"Brave Search rejected the API key (HTTP {status})", retryable=False, remedy=_AUTH_REMEDY)
    if status == 402 or code == "QUOTA_LIMITED" or (status == 429 and _monthly_quota_spent(headers)):
        return SearchError(f"Brave Search quota exceeded (HTTP {status})", retryable=False, remedy=_QUOTA_REMEDY)
    if status == 429:
        return SearchError(
            "Brave Search rate limit hit (HTTP 429)",
            retryable=True,
            remedy="Wait a second between searches (the smallest plans allow 1 request per second).",
        )
    if status >= 500:
        return SearchError(f"Brave Search answered HTTP {status}", retryable=True)
    detail = error.get("detail") if isinstance(error, dict) else None
    return SearchError(f"Brave Search answered HTTP {status}" + (f": {detail}" if detail else ""), retryable=False)


def brave_doctor_checks(provider: BraveProvider) -> list[Check]:
    """Connect only: no query and no key is sent, so nothing counts against the quota."""
    try:
        probe_connection(provider.endpoint)
    except OSError as exc:
        return [
            Check(
                "web search reachable",
                False,
                f"Brave Search at {provider.endpoint.netloc} not reachable: {exc} (connection only, no query sent)",
                hint="check the network connection",
            )
        ]
    return [
        Check(
            "web search reachable",
            True,
            f"Brave Search at {provider.endpoint.netloc} reachable (connection only, no query sent)",
        )
    ]
