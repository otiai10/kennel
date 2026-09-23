"""LlamaServerProvider: a llama.cpp ``llama-server`` over its OpenAI-compatible HTTP API.

The user starts and owns the server; Kennel never downloads a model nor launches a
process. The tool loop is the shared one (:mod:`kennel.providers.chatloop`) -- this
module is only a transport.

Design notes (measured against llama-server 0.4.1, build b10964, Qwen3-4B-GGUF
Q4_K_M, ``-c 32768 --jinja``, 2026-09-18):

* Requests are always streamed, with ``stream_options.include_usage`` so the final
  ``choices: []`` chunk carries the token counts. Those counts are the only numbers
  ``usage()`` reports (principle 4).
* A streamed tool call arrives as ``delta.tool_calls`` fragments: the first carries
  ``id`` and ``function.name``, the rest only ``index`` and a slice of
  ``function.arguments``. The first chunk of any completion may carry
  ``content: null``.
* A thinking model's scratchpad comes back separately as
  ``delta.reasoning_content`` and is dropped here, so it never reaches the answer.
  It is still generated, so ``--reasoning off`` on the server is worth it.
* ``tool_choice: "none"`` is not used to end a turn: it leaks raw
  ``<tool_call>`` text into ``content``.
* Overflowing the window is HTTP 400 with ``error.type ==
  "exceed_context_size_error"``, which maps to :class:`ContextLimitError` so
  ``Session.run``'s compact-and-retry applies unchanged.
* A server started with ``--api-key`` (measured 2026-09-23) answers ``/props``,
  ``/v1/models`` and ``POST /v1/chat/completions`` with HTTP 401 and
  ``{"error": {"message": "Invalid API Key", "type": "authentication_error"}}`` both
  without a key and with a wrong one; only ``/health`` stays open. ``api_key`` is sent as
  ``Authorization: Bearer`` on every request, and a 401 becomes a fixed message that says
  whether a key was sent: the server's body is dropped so nothing it says reaches events
  or the session log. The key comes from the environment (``KENNEL_LLAMA_SERVER_API_KEY``
  by default, see the registry), never from the config file.

Threading: ``http.client`` is blocking, so every request runs on a worker thread
that hands parsed SSE events to the caller's loop through an
:class:`asyncio.Queue` -- the same separation ``providers/apple.py`` uses for the
SDK's blocking calls. Cancelling the turn shuts the socket down, which unblocks
that thread's read; each request uses its own connection, so an aborted one is
never reused.
"""

from __future__ import annotations

import asyncio
import dataclasses
import http.client
import ipaddress
import json
import socket
import threading
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ..diagnostics import Check
from ..errors import (
    ConfigurationError,
    ContextLimitError,
    ModelUnavailableError,
    ProviderError,
)
from ..tools.base import Tool
from .base import ModelProvider, ProviderInfo, ProviderSession, ToolInvoker, Usage
from .chatloop import ChatChunk, ChatLoopSession, ChatMessage, ToolCallDelta

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
API_KEY_ENV = "KENNEL_LLAMA_SERVER_API_KEY"  # the default declared in providers/registry.py
DEFAULT_TIMEOUT = 300.0

START_HINT = (
    "Start one first, for example: llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M --jinja "
    "-c 32768 --port 8080 --host 127.0.0.1 (brew install llama.cpp). Kennel does not "
    "download models or start the server itself. Point elsewhere with "
    'providers.llama-server.base_url in kennel.json.'
)

_KEY_HINT = (
    f"Set {API_KEY_ENV} to the server's --api-key, or name another variable in "
    "providers.llama-server.secrets.api_key; `kennel doctor` shows which variable is read."
)


# -- the endpoint ------------------------------------------------------------


@dataclass(frozen=True)
class _Endpoint:
    scheme: str
    host: str
    port: int
    prefix: str  # path prefix from the base URL, "" in the usual case
    api_key: str | None = field(default=None, repr=False)  # never in a repr

    def connection(self, timeout: float) -> http.client.HTTPConnection:
        factory = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return factory(self.host, self.port, timeout=timeout)

    def path(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"

    def headers(self, base: dict[str, str]) -> dict[str, str]:
        """``base`` plus ``Authorization: Bearer`` when there is a key."""
        return {**base, "Authorization": f"Bearer {self.api_key}"} if self.api_key else base

    def http_error(self, status: int, raw: bytes) -> BaseException:
        return _http_error(status, raw, key_sent=bool(self.api_key))

    @property
    def sends_key_in_clear(self) -> bool:
        """A key over plain http to another machine can be read on the way."""
        return bool(self.api_key) and self.scheme == "http" and not self.is_loopback

    @property
    def is_loopback(self) -> bool:
        """Does this address stay on the machine? (principle 1: what ``mode`` declares)"""
        host = self.host.lower()
        if host == "localhost" or host.endswith(".localhost"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False


def _parse_base_url(base_url: str, api_key: str | None = None) -> _Endpoint:
    parts = urlsplit(base_url if "//" in base_url else f"//{base_url}", scheme="http")
    if parts.hostname is None:
        raise ConfigurationError(f"llama-server: base_url {base_url!r} has no host")
    scheme = parts.scheme if parts.scheme in ("http", "https") else "http"
    return _Endpoint(
        scheme=scheme,
        host=parts.hostname,
        port=parts.port or (443 if scheme == "https" else 80),
        prefix=parts.path.rstrip("/"),
        api_key=api_key or None,
    )


# -- errors ------------------------------------------------------------------


def _http_error(status: int, raw: bytes, *, key_sent: bool = False) -> BaseException:
    """Map a non-200 response to the error a Kennel consumer should see.

    A 401 gets a fixed message, without the server's body: all the provider knows is
    whether it sent a key, not which variable the key came from.
    """
    if status == 401:
        what = (
            "llama-server rejected the API key that was sent (HTTP 401)."
            if key_sent
            else "llama-server refused the request (HTTP 401): no API key was sent."
        )
        return ProviderError(f"{what} {_KEY_HINT}", status=401)
    text = raw.decode("utf-8", "replace")
    error: Any = None
    try:
        error = json.loads(text).get("error")
    except (ValueError, AttributeError):
        error = None
    if not isinstance(error, dict):
        error = {}
    message = error.get("message") or text.strip()[:400] or "(no body)"
    if status == 400 and error.get("type") == "exceed_context_size_error":
        return ContextLimitError(f"The request exceeded llama-server's context size: {message}")
    return ProviderError(f"llama-server returned HTTP {status}: {message}")


def _transport_error(exc: BaseException) -> BaseException:
    """Map a socket-level failure. Unreachable means unavailable, not broken."""
    if isinstance(exc, TimeoutError):  # a subclass of OSError, so it goes first
        return ProviderError(f"llama-server did not answer within the timeout: {exc}")
    if isinstance(exc, OSError):
        return ModelUnavailableError(f"Cannot reach llama-server: {exc}. {START_HINT}")
    if isinstance(exc, Exception):
        return ProviderError(f"The llama-server request failed: {exc}")
    return exc


# -- HTTP --------------------------------------------------------------------


def _get_json(endpoint: _Endpoint, path: str, timeout: float) -> dict[str, Any]:
    conn = endpoint.connection(timeout)
    try:
        conn.request("GET", endpoint.path(path), headers=endpoint.headers({"Accept": "application/json"}))
        response = conn.getresponse()
        raw = response.read()
    except OSError as exc:
        raise _transport_error(exc) from exc
    finally:
        conn.close()
    if response.status != 200:
        raise endpoint.http_error(response.status, raw)
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError as exc:
        raise ProviderError(f"llama-server sent invalid JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"llama-server sent an unexpected {path} payload")
    return data


async def _post_sse(
    endpoint: _Endpoint, path: str, body: dict[str, Any], timeout: float
) -> AsyncIterator[dict[str, Any]]:
    """POST ``body`` and yield each parsed ``data:`` event on the caller's loop.

    The request runs on a worker thread. Leaving this iterator (normally, or because
    the turn was cancelled) shuts the socket down, which makes the thread's blocked
    read return so it can close its own connection. The shutdown has to be a socket
    call rather than ``response.close()``: the buffered reader is locked by the
    thread sitting in that read, so closing it here would block the event loop for
    as long as the request would have taken.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | BaseException | None] = asyncio.Queue()
    leaving = threading.Event()
    sock: socket.socket | None = None

    def put(item: dict[str, Any] | BaseException | None) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # pragma: no cover - the caller's loop is gone
            pass

    def worker() -> None:
        nonlocal sock
        if leaving.is_set():  # cancelled before the request was even made
            put(None)
            return
        conn = endpoint.connection(timeout)
        response = None
        try:
            conn.request(
                "POST",
                endpoint.path(path),
                json.dumps(body),
                endpoint.headers({"Content-Type": "application/json", "Accept": "text/event-stream"}),
            )
            # Taken now, not from the finally below: a response that closes the
            # connection takes the socket with it and leaves conn.sock as None.
            sock = conn.sock
            response = conn.getresponse()
            if response.status != 200:
                put(endpoint.http_error(response.status, response.read()))
                return
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue  # comments, event: lines and keep-alive blanks
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    return
                try:
                    put(json.loads(payload))
                except ValueError as exc:
                    put(ProviderError(f"llama-server sent a malformed stream chunk: {exc}"))
                    return
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            if not leaving.is_set():  # a read that failed because we closed the socket
                put(_transport_error(exc))
        finally:
            put(None)
            if response is not None:
                response.close()  # safe on this thread: it is the one that was reading
            conn.close()

    threading.Thread(target=worker, name="kennel-llama-http", daemon=True).start()
    try:
        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        leaving.set()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:  # pragma: no cover - the request already finished
                pass


# -- the transport -----------------------------------------------------------


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    return Usage(input_tokens=raw.get("prompt_tokens"), output_tokens=raw.get("completion_tokens"))


def _to_chunk(event: dict[str, Any]) -> ChatChunk:
    """Translate one streamed event. ``delta.reasoning_content`` is deliberately dropped."""
    usage = _usage(event.get("usage"))
    choices = event.get("choices") or []
    if not choices:
        return ChatChunk(usage=usage)  # the include_usage chunk carries no choices
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get("delta") or {}
    deltas = [
        ToolCallDelta(
            index=int(raw.get("index", position)),
            id=raw.get("id"),
            name=(raw.get("function") or {}).get("name"),
            arguments=(raw.get("function") or {}).get("arguments") or "",
        )
        for position, raw in enumerate(delta.get("tool_calls") or [])
        if isinstance(raw, dict)
    ]
    return ChatChunk(delta.get("content") or "", deltas, choice.get("finish_reason"), usage)


class LlamaServerTransport:
    """Turns one :class:`~kennel.providers.chatloop.ChatTransport` request into HTTP."""

    def __init__(
        self,
        endpoint: _Endpoint,
        *,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout
        self._extra_body = extra_body

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        schema: dict[str, Any] | None,
    ) -> AsyncIterator[ChatChunk]:
        body: dict[str, Any] = {
            "messages": [message.to_dict() for message in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._model:
            body["model"] = self._model
        if tools:
            body["tools"] = list(tools)
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        if self._extra_body:
            body.update(self._extra_body)  # last, so a caller can override anything
        async for event in _post_sse(self._endpoint, "/v1/chat/completions", body, self._timeout):
            yield _to_chunk(event)

    async def close(self) -> None:
        return None  # every request owns its connection


# -- the provider ------------------------------------------------------------


class LlamaServerProvider(ModelProvider):
    """Talk to a running ``llama-server``.

    ``base_url`` decides whether inference stays on this machine: a loopback host
    reports ``mode: "local"``, anything else ``"remote"`` (principle 1). ``model``
    overrides the name the server reports, ``timeout`` bounds each socket
    operation, and ``extra_body`` is merged into every request body for
    model-specific knobs such as ``chat_template_kwargs``; Kennel adds none itself.
    ``api_key`` is sent as ``Authorization: Bearer`` on every request; the registry
    passes it from the environment, and no repr or message carries it.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        extra_body: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url
        self._endpoint = _parse_base_url(base_url, api_key)
        self._model = model
        self._timeout = float(timeout)
        self._extra_body = dict(extra_body) if extra_body else None
        self._probed = False
        self.info = ProviderInfo(
            name="llama-server",
            model=model or f"llama-server at {base_url}",
            mode="local" if self._endpoint.is_loopback else "remote",
            context_window_tokens=None,
        )

    def check_availability(self) -> None:
        """Ask the server what it is running, and fill in :attr:`info` from the answer."""
        props = _get_json(self._endpoint, "/props", self._timeout)
        models = _get_json(self._endpoint, "/v1/models", self._timeout)
        window = (props.get("default_generation_settings") or {}).get("n_ctx")
        entries = [entry for entry in (models.get("models") or []) if isinstance(entry, dict)]
        reported = entries[0].get("name") if entries else None
        self._model = self._model or reported
        self.info = dataclasses.replace(
            self.info,
            model=self._model or self.info.model,
            context_window_tokens=int(window) if isinstance(window, int) else None,
        )
        self._probed = True

    async def create_session(
        self, *, instructions: str, tools: Sequence[Tool], invoke: ToolInvoker
    ) -> ProviderSession:
        if not self._probed:
            # Probe once so info (model, window, and therefore context_usage) is honest
            # even for an SDK caller that never asked check_availability() itself.
            self.check_availability()
        transport = LlamaServerTransport(
            self._endpoint, model=self._model, timeout=self._timeout, extra_body=self._extra_body
        )
        return ChatLoopSession(transport, instructions=instructions, tools=tools, invoke=invoke)


# -- registry hooks (see kennel.providers.registry) --------------------------


def llama_server_provider(**options: Any) -> LlamaServerProvider:
    return LlamaServerProvider(**options)


def llama_server_doctor_checks(**options: Any) -> list[Check]:
    """The ``kennel doctor`` checks for a llama-server: is it there, and what is it running?

    A 401 is raised rather than turned into a check: the hint for it has to name the
    variable the key is actually read from, which only ``kennel doctor`` knows (the
    provider is given the key, never where it came from). The checks made before it
    (the plain-http warning) travel on the exception as ``checks``.
    """
    try:
        provider = LlamaServerProvider(**options)
    except (ConfigurationError, TypeError) as exc:
        return [
            Check("llama-server", False, f"llama-server options: {exc}", hint="fix providers.llama-server in the config"),
            Check("model", False, "skipped: the provider could not be built"),
        ]
    warnings = []
    if provider._endpoint.sends_key_in_clear:
        warnings.append(
            Check(
                "llama-server key transport",
                True,
                f"warning: the API key is sent unencrypted to {provider._endpoint.host} over http://; "
                "use https:// unless the network is trusted",
            )
        )
    try:
        provider.check_availability()
    except ProviderError as exc:
        if exc.status == 401:
            exc.checks = warnings  # type: ignore[attr-defined]  # kennel doctor shows them first
            raise
        return [
            *warnings,
            Check("llama-server", False, str(exc), hint=START_HINT),
            Check("model", False, f"skipped: {provider.base_url} is not reachable"),
        ]
    info = provider.info
    window = f"{info.context_window_tokens} tokens" if info.context_window_tokens else "not declared"
    return [
        *warnings,
        Check("llama-server", True, f"llama-server at {provider.base_url} reachable (mode: {info.mode})"),
        Check("model", True, f"{info.model} (context window: {window})"),
    ]
