"""The one HTTP GET every built-in search provider uses: bounded in time and in bytes.

Each provider only builds its request and parses the response; this module owns the rest,
so the limits cannot differ between providers:

* ``http.client`` is blocking, so the request runs on a worker thread and the caller's
  event loop stays free (the same separation ``providers/llama_server.py`` uses);
* ``timeout`` is a deadline for the whole request, not only for each socket operation, so
  a server that trickles bytes cannot hold a turn open;
* cancelling the awaiting task (``Session.interrupt()``, a turn timeout) shuts the socket
  down, which unblocks the worker's read so the request really stops;
* the body is read in chunks and reading stops once it passes ``max_bytes``: an oversized
  answer fails with :class:`~kennel.errors.SearchError` instead of being held in memory.

Redirects are not followed and no compression is requested (``Accept-Encoding`` is left
unset); both keep the helper small and neither is needed by the built-in providers.
"""

from __future__ import annotations

import asyncio
import http.client
import socket
import ssl
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

from ..errors import ConfigurationError, SearchError

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_BYTES = 1_000_000
_CHUNK = 64 * 1024


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]  # lower-cased names
    body: bytes


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
    loop = asyncio.get_running_loop()
    done: asyncio.Future[HttpResponse] = loop.create_future()
    leaving = threading.Event()
    sock: socket.socket | None = None
    target = f"{endpoint.path}{path}?{urlencode(params)}"
    request_headers = {"Accept": "application/json", "User-Agent": "kennel", **(headers or {})}

    def settle(result: HttpResponse | BaseException) -> None:
        def apply() -> None:
            if done.done():
                return
            if isinstance(result, BaseException):
                done.set_exception(result)
            else:
                done.set_result(result)

        try:
            loop.call_soon_threadsafe(apply)
        except RuntimeError:  # pragma: no cover - the caller's loop is gone
            pass

    def worker() -> None:
        nonlocal sock
        conn = endpoint.connection(timeout)
        response = None
        try:
            if leaving.is_set():
                return
            conn.request("GET", target, headers=request_headers)
            sock = conn.sock
            if leaving.is_set():  # cancelled while connecting: the socket was not known yet
                return
            response = conn.getresponse()
            length = response.getheader("Content-Length")
            if length is not None and length.isdigit() and int(length) > max_bytes:
                settle(_too_large(service, max_bytes))
                return
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    settle(_too_large(service, max_bytes))
                    return
                chunks.append(chunk)
            names = {k.lower(): v for k, v in response.getheaders()}
            settle(HttpResponse(response.status, names, b"".join(chunks)))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            if not leaving.is_set():  # otherwise the read failed because we shut the socket
                settle(_transport_error(service, endpoint, exc, timeout))
        finally:
            if response is not None:
                response.close()
            conn.close()

    threading.Thread(target=worker, name="kennel-search-http", daemon=True).start()
    try:
        return await asyncio.wait_for(done, timeout)
    except asyncio.TimeoutError:
        raise SearchError(f"{service} did not answer within {timeout:g} s", retryable=True) from None
    finally:
        leaving.set()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:  # the request had already finished
                pass


def probe_connection(endpoint: Endpoint, timeout: float = 5.0) -> None:
    """Open (and close) a connection to ``endpoint`` without sending a request.

    For ``kennel doctor``: it shows the service is reachable without sending a query, a key
    or anything that counts against a quota. Raises ``OSError`` when it is not.
    """
    with socket.create_connection((endpoint.host, endpoint.port), timeout=timeout) as raw:
        if endpoint.scheme == "https":
            with ssl.create_default_context().wrap_socket(raw, server_hostname=endpoint.host):
                pass
