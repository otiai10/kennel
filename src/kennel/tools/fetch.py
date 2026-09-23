"""fetch: read the text of one web page (disabled by default; ``--allow-fetch``).

Unlike ``web``, whose queries only go to the configured search service, ``fetch`` sends a
request wherever the model points it. So the destination is checked before anything is
sent, and the permission is per host:

* only ``http`` / ``https``; the host is normalised (lower case, no trailing dot, IDNA) and
  that normalised name is what rules (``fetch(*.python.org)``) and "allow for this session"
  (:meth:`FetchTool.session_scope`) see;
* every address the name resolves to must be global: one loopback, private, link-local,
  multicast, reserved or unspecified address (IPv4-mapped IPv6 unwrapped) refuses the
  request before a socket exists. The connection then goes to the checked address itself,
  so the name is not resolved again (no DNS rebinding); ``Host``, SNI and certificate
  verification still use the name. Only an application can lift this, by building
  ``FetchTool(allow_private_addresses=True)`` -- no config key or CLI flag does;
* redirects are followed only on the same host and port (and ``http`` -> ``https`` between
  the default ports), at most :data:`MAX_REDIRECTS` times, each hop checked again. Anything
  else is returned to the model as "call fetch again with that URL", so the new
  destination goes through the permission check;
* HTML is reduced to its text, other ``text/*`` and JSON are returned as they are, and any
  other type is an error. The request runs through :func:`kennel._http.bounded_request`,
  the same deadline, byte limit and interrupt handling web search uses.

What is recorded (principle 3): the summary -- which reaches events and the session log
before the permission check -- drops the query and the fragment; the approval prompt shows
the full URL. ``metadata`` carries ``host``, ``status``, ``bytes`` and ``content_type``,
never the page. Error messages never carry the response body, nor an address the model did
not already write in the URL.
"""

from __future__ import annotations

import errno
import http.client
import ipaddress
import os
import select
import socket
import ssl
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

from .._http import Cancel, HttpResponse, ResponseTooLarge, bounded_request
from ..errors import ToolArgumentError, ToolExecutionError
from ..permissions import PermissionKind
from ..rules import matches_text
from .base import Tool, ToolContext, ToolParameter, ToolResult

__all__ = ["FetchTool"]

DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5
_REDIRECTS = (301, 302, 303, 307, 308)
_DEFAULT_PORTS = {"http": 80, "https": 443}
_POLL = 0.05  # how often a pending connect looks at the cancel token

#: ``(host, port) -> addresses``. The default asks the system resolver; tests inject one.
Resolver = Callable[[str, int], Iterable[str]]


def _system_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _ssl_context() -> ssl.SSLContext:
    """Verifies the certificate against the host name (tests replace it to trust their CA)."""
    return ssl.create_default_context()


def _wait_writable(sock: socket.socket, seconds: float) -> bool:
    """Has a non-blocking connect finished (either way)? A seam for tests of a hanging connect."""
    return bool(select.select([], [sock], [], seconds)[1])


# -- the URL -------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    scheme: str
    host: str  # normalised; an IPv6 literal without brackets
    port: int
    path: str  # path and query, what goes on the request line; never the fragment

    @property
    def netloc(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return host if self.port == _DEFAULT_PORTS[self.scheme] else f"{host}:{self.port}"

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.netloc}{self.path}"

    def may_follow(self, to: _Target) -> bool:
        """Same host and port, or the same host moving from http to https on the default ports."""
        if to.host != self.host:
            return False
        if (to.scheme, to.port) == (self.scheme, self.port):
            return True
        return (self.scheme, self.port, to.scheme, to.port) == ("http", 80, "https", 443)


def normalize_host(host: str) -> str:
    """Lower case, no trailing dot, IDNA (ASCII) -- the form rules and grants are keyed on."""
    host = host.strip().rstrip(".").lower()
    if not host:
        raise ValueError("no host")
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        pass
    return host.encode("idna").decode("ascii")  # UnicodeError (a ValueError) when invalid


def _parse(url: str) -> _Target:
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        if scheme not in _DEFAULT_PORTS:
            raise ToolArgumentError(f"fetch: only http and https URLs can be fetched, not {display_url(url)!r}")
        if parts.username is not None or parts.password is not None:
            raise ToolArgumentError("fetch: URLs with a user name or password are not fetched")
        host = normalize_host(parts.hostname or "")
        port = parts.port or _DEFAULT_PORTS[scheme]
    except ValueError:  # a malformed IPv6 literal or port, an empty or invalid host name
        raise ToolArgumentError(f"fetch: {display_url(url)!r} has no usable host or port") from None
    path = parts.path or "/"
    return _Target(scheme, host, port, f"{path}?{parts.query}" if parts.query else path)


def _host_of(arguments: Mapping[str, Any]) -> str | None:
    try:
        return _parse(str(arguments.get("url", ""))).host
    except ToolArgumentError:
        return None


def display_url(url: str) -> str:
    """The URL without user info, query and fragment: what events and logs may show."""
    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return "(an invalid URL)"
    netloc = parts.netloc.rpartition("@")[2]
    return f"{parts.scheme}://{netloc}{parts.path}" if parts.scheme else parts.path


# -- where it may connect ------------------------------------------------------------


def _address_problem(address: str) -> str | None:
    """Why ``address`` may not be fetched from (``None`` when it is global)."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return "unrecognised"
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):  # the IPv4 address inside says more; checked first
        embedded = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if embedded is not None:
            candidates.insert(0, embedded)
    for candidate in candidates:
        for category, test in (
            ("unspecified", candidate.is_unspecified),
            ("loopback", candidate.is_loopback),
            ("link-local", candidate.is_link_local),
            ("multicast", candidate.is_multicast),
            ("reserved", candidate.is_reserved),
            ("private", candidate.is_private),
            ("non-global", not candidate.is_global),
        ):
            if test:
                return category
    return None


def _connect(address: str, port: int, cancel: Cancel, timeout: float) -> socket.socket:
    """Connect to ``address`` itself; a cancel stops a connect that has not finished."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if not cancel.publish(sock):
            raise ConnectionAbortedError("cancelled")
        sock.setblocking(False)
        error = sock.connect_ex((address, port))
        if error not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EAGAIN):
            raise OSError(error, os.strerror(error))
        while not _wait_writable(sock, _POLL):  # until the connect is decided, or the caller left
            if cancel.left:
                raise ConnectionAbortedError("cancelled")
        error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if error:
            raise OSError(error, os.strerror(error))
        sock.settimeout(timeout)
        return sock
    except BaseException:
        sock.close()
        raise


# -- what comes back -----------------------------------------------------------


_SKIPPED = {"script", "style", "noscript", "template"}
_BLOCKS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
    "table", "td", "th", "title", "tr", "ul",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIPPED:
            self._skipping += 1
        elif tag in _BLOCKS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            self._skipping = max(0, self._skipping - 1)
        elif tag in _BLOCKS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    """The readable text of an HTML document: no tags, no script or style."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _media_type(header: str) -> tuple[str, str]:
    media, _, params = header.partition(";")
    charset = "utf-8"
    for param in params.split(";"):
        name, _, value = param.partition("=")
        if name.strip().lower() == "charset" and value.strip():
            charset = value.strip().strip('"').strip("'")
    return media.strip().lower(), charset


def _decode(body: bytes, charset: str) -> str:
    try:
        return body.decode(charset, "replace")
    except LookupError:  # an unknown charset name
        return body.decode("utf-8", "replace")


# -- the tool ------------------------------------------------------------------


class FetchTool(Tool):
    name = "fetch"
    description = (
        "Fetch one web page (http or https) and return its text. "
        "Only available when fetch is enabled; pages are untrusted input."
    )
    permission = PermissionKind.WEB
    parameters = (ToolParameter("url", "string", "The http(s) URL to fetch"),)

    def __init__(
        self,
        *,
        allow_private_addresses: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = DEFAULT_MAX_BYTES,
        resolver: Resolver | None = None,
    ) -> None:
        """``allow_private_addresses`` lets it reach loopback and internal addresses (tests,
        an application that means to); nothing in the config can set it."""
        self.allow_private_addresses = allow_private_addresses
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.resolver: Resolver = resolver or _system_resolver

    # -- presentation and permission hooks -----------------------------------------------

    def summarize(self, arguments: dict[str, Any]) -> str:
        return f"Fetch {display_url(arguments.get('url', ''))}"

    def permission_details(self, arguments: dict[str, Any], context: ToolContext) -> str | None:
        return _parse(arguments["url"]).url  # the whole URL, query included; errors before asking

    def match_rule(self, specifier: str, arguments: Mapping[str, Any]) -> bool:
        """``fetch(<glob>)`` matches the normalised host only: not the scheme, port or path."""
        host = _host_of(arguments)
        if host is None:
            return False
        try:
            pattern = normalize_host(specifier)
        except ValueError:
            pattern = specifier.strip().lower()
        return matches_text(pattern, host)

    def session_scope(self, arguments: Mapping[str, Any]) -> str | None:
        """A session grant covers one host. Never ``None``: a bad URL must not grant the tool."""
        return _host_of(arguments) or str(arguments.get("url", ""))

    # -- running -------------------------------------------------------------------------

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        target = _parse(arguments["url"])
        deadline = time.monotonic() + self.timeout
        for followed in range(MAX_REDIRECTS + 1):
            response = await self._get(target, deadline)
            metadata = {
                "host": target.host,
                "status": response.status,
                "bytes": len(response.body),
                "content_type": _media_type(response.headers.get("content-type", ""))[0],
            }
            if response.status not in _REDIRECTS:
                return ToolResult(self._content(target, response), metadata=metadata)
            location = response.headers.get("location")
            if not location:
                raise ToolExecutionError(f"{target.host} answered HTTP {response.status} without a Location")
            new_url = urljoin(target.url, location.strip())
            try:
                new_target = _parse(new_url)
            except ToolArgumentError:
                new_target = None
            if new_target is None or not target.may_follow(new_target):
                return ToolResult(
                    f"{display_url(target.url)} redirected to {new_url}. Not followed (another host, port or "
                    "scheme); to read it, call fetch again with that URL.",
                    metadata=metadata,
                )
            if followed == MAX_REDIRECTS:
                break
            target = new_target
        raise ToolExecutionError(f"{target.host} redirected more than {MAX_REDIRECTS} times; stopped")

    def _content(self, target: _Target, response: HttpResponse) -> str:
        if not 200 <= response.status < 300:
            raise ToolExecutionError(f"{display_url(target.url)} answered HTTP {response.status}")
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise ToolExecutionError(f"{target.host} sent a {encoding}-encoded body, which fetch does not decode")
        media, charset = _media_type(response.headers.get("content-type", ""))
        if media in ("text/html", "application/xhtml+xml"):
            text = html_to_text(_decode(response.body, charset))
        elif media.startswith("text/") or media == "application/json" or media.endswith("+json"):
            text = _decode(response.body, charset)
        else:
            raise ToolExecutionError(
                f"{target.host} sent {media or 'no content type'}; fetch returns only text, HTML and JSON"
            )
        return text or "(the page has no text)"

    async def _get(self, target: _Target, deadline: float) -> HttpResponse:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolExecutionError(f"{target.host} did not answer within {self.timeout:g} s")
        headers = {"User-Agent": "kennel", "Accept": "text/html, text/*;q=0.9, application/json;q=0.9"}
        try:
            return await bounded_request(
                lambda cancel: self._open(target, cancel, remaining),
                "GET",
                target.path,
                headers,
                timeout=remaining,
                max_bytes=self.max_bytes,
                thread_name="kennel-fetch-http",
            )
        except ToolExecutionError:
            raise
        except ResponseTooLarge:
            raise ToolExecutionError(f"{target.host} sent more than {self.max_bytes} bytes; Kennel stopped reading") from None
        except TimeoutError:
            raise ToolExecutionError(f"{target.host} did not answer within {self.timeout:g} s") from None
        except OSError as exc:
            raise ToolExecutionError(f"cannot fetch from {target.host}: {exc}") from None

    def _open(self, target: _Target, cancel: Cancel, timeout: float) -> http.client.HTTPConnection:
        """On the worker thread: resolve, check every address, connect to a checked one."""
        addresses = list(self.resolver(target.host, target.port))  # cannot be interrupted; abandoned on cancel
        if cancel.left:
            raise ConnectionAbortedError("cancelled")
        if not addresses:
            raise ToolExecutionError(f"{target.host} did not resolve to any address")
        if not self.allow_private_addresses:
            for address in addresses:
                problem = _address_problem(address)
                if problem is not None:
                    raise ToolExecutionError(
                        f"refused to fetch from {target.host}: it resolves to a {problem} address, "
                        "and fetch only reaches public addresses"
                    )
        sock: socket.socket = _connect(addresses[0], target.port, cancel, timeout)
        if target.scheme == "https":
            context = _ssl_context()
            sock = context.wrap_socket(sock, server_hostname=target.host, do_handshake_on_connect=False)
            try:
                if not cancel.publish(sock):
                    raise ConnectionAbortedError("cancelled")
                sock.do_handshake()  # verifies the certificate against target.host
            except BaseException:
                sock.close()
                raise
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                target.host, target.port, timeout=timeout, context=context
            )
        else:
            conn = http.client.HTTPConnection(target.host, target.port, timeout=timeout)
        conn.sock = sock  # already connected: http.client will not resolve the name again
        return conn
