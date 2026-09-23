"""The one HTTP GET every built-in search provider uses: bounded in time and in bytes.

Each provider only builds its request and parses the response. The request itself -- the
worker thread, the deadline for the whole request, stopping on cancel, the ``max_bytes``
cut-off -- is :func:`kennel._http.bounded_request`, shared with the ``fetch`` tool so the
limits cannot differ between them. This module turns its failures into
:class:`~kennel.errors.SearchError`: an oversized answer fails instead of being held in memory.

Redirects are not followed and no compression is requested (``Accept-Encoding`` is left
unset); both keep the helper small and neither is needed by the built-in providers.
"""

from __future__ import annotations

import http.client
import socket
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

from .._http import HttpResponse, ResponseTooLarge, bounded_request
from ..errors import ConfigurationError, SearchError

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_BYTES = 1_000_000


@dataclass(frozen=True)
class Endpoint:
    """A parsed base URL: what to connect to and how to show it."""

    scheme: str
    host: str
    port: int
    path: str  # without a trailing slash

    @classmethod
    def parse(cls, url: str, *, what: str) -> Endpoint:
        parts = urlsplit(url if "//" in url else f"//{url}", scheme="http")
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigurationError(f"{what}: {url!r} is not an http(s) URL")
        default_port = 443 if parts.scheme == "https" else 80
        return cls(parts.scheme, parts.hostname, parts.port or default_port, parts.path.rstrip("/"))

    @property
    def netloc(self) -> str:
        default_port = 443 if self.scheme == "https" else 80
        return self.host if self.port == default_port else f"{self.host}:{self.port}"

    def connection(self, timeout: float) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)


def _too_large(service: str, max_bytes: int) -> SearchError:
    return SearchError(
        f"{service} sent more than {max_bytes} bytes; Kennel stopped reading",
        retryable=False,
        remedy="Raise max_response_bytes for this search provider in the config if that size is expected.",
    )


def _transport_error(service: str, endpoint: Endpoint, exc: BaseException, timeout: float) -> SearchError:
    if isinstance(exc, TimeoutError):  # a subclass of OSError, so it goes first
        return SearchError(f"{service} did not answer within {timeout:g} s", retryable=True)
    if isinstance(exc, OSError):
        return SearchError(
            f"cannot reach {service} at {endpoint.netloc}: {exc}",
            retryable=True,
            remedy="Check that the search service is running and reachable.",
        )
    return SearchError(f"the request to {service} failed: {exc}", retryable=True)


async def http_get(
    endpoint: Endpoint,
    path: str,
    params: Mapping[str, str | int],
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    service: str,
) -> HttpResponse:
    """GET ``endpoint.path + path`` with ``params`` and return the (bounded) response.

    Raises :class:`SearchError` on a transport failure, the deadline or an oversized body.
    ``service`` names the search service in those messages.
    """
    target = f"{endpoint.path}{path}?{urlencode(params)}"
    request_headers = {"Accept": "application/json", "User-Agent": "kennel", **(headers or {})}
    try:
        return await bounded_request(
            lambda _cancel: endpoint.connection(timeout),  # connects on the request
            "GET",
            target,
            request_headers,
            timeout=timeout,
            max_bytes=max_bytes,
            thread_name="kennel-search-http",
        )
    except ResponseTooLarge:
        raise _too_large(service, max_bytes) from None
    except Exception as exc:  # noqa: BLE001 - every transport failure becomes a SearchError
        raise _transport_error(service, endpoint, exc, timeout) from exc


def probe_connection(endpoint: Endpoint, timeout: float = 5.0) -> None:
    """Open (and close) a connection to ``endpoint`` without sending a request.

    For ``kennel doctor``: it shows the service is reachable without sending a query, a key
    or anything that counts against a quota. Raises ``OSError`` when it is not.
    """
    with socket.create_connection((endpoint.host, endpoint.port), timeout=timeout) as raw:
        if endpoint.scheme == "https":
            with ssl.create_default_context().wrap_socket(raw, server_hostname=endpoint.host):
                pass
