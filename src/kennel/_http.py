"""The one bounded HTTP request both network tools use: web search and ``fetch``.

The callers only open a connection and interpret the answer; this module owns the rest, so
the limits cannot differ between them:

* ``http.client`` is blocking, so the request runs on a worker thread and the caller's
  event loop stays free (the same separation ``providers/llama_server.py`` uses);
* ``timeout`` is a deadline for the whole request, not only for each socket operation, so
  a server that trickles bytes cannot hold a turn open;
* cancelling the awaiting task (``Session.interrupt()``, a turn timeout) marks the
  :class:`Cancel` token and shuts its socket down, which unblocks the worker so the request
  really stops. Once the token is marked, no request is sent: the check before sending and
  the cancel take the same lock;
* the body is read in chunks and reading stops once it passes ``max_bytes``.

Errors are raised as they are (:class:`ResponseTooLarge`, :class:`TimeoutError`, whatever
the connection raised); each caller turns them into its own error type and wording.
"""

from __future__ import annotations

import asyncio
import http.client
import socket
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

_CHUNK = 64 * 1024


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]  # lower-cased names
    body: bytes


class ResponseTooLarge(Exception):
    """The body passed ``max_bytes``; reading stopped there."""


class Cancel:
    """Shared between the caller and the worker: has the caller left, and which socket to stop.

    The opener publishes the socket as soon as it exists (before connecting, if it can), so
    a cancel can stop a connect or a read; the worker asks :meth:`may_send` right before the
    request goes out.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._left = False
        self._sock: socket.socket | None = None

    @property
    def left(self) -> bool:
        with self._lock:
            return self._left

    def publish(self, sock: socket.socket | None) -> bool:
        """Remember ``sock`` so :meth:`leave` can stop it. ``False`` if the caller already left."""
        with self._lock:
            if self._left:
                return False
            self._sock = sock
            return True

    def may_send(self) -> bool:
        """``True`` unless the caller left; checked under the lock :meth:`leave` takes."""
        with self._lock:
            return not self._left

    def leave(self) -> None:
        with self._lock:
            self._left = True
            sock = self._sock
        if sock is not None:
            try:
                # The plain socket's shutdown, also for an SSLSocket: it only has to unblock
                # the worker, whatever state the TLS layer is in.
                socket.socket.shutdown(sock, socket.SHUT_RDWR)
            except OSError:  # not connected yet, or already finished
                pass


Opener = Callable[[Cancel], http.client.HTTPConnection]


async def bounded_request(
    open_connection: Opener,
    method: str,
    target: str,
    headers: Mapping[str, str],
    *,
    timeout: float,
    max_bytes: int,
    thread_name: str,
) -> HttpResponse:
    """Send one request on the connection ``open_connection`` returns; read a bounded answer.

    ``open_connection`` runs on the worker thread. It may connect itself (publishing the
    socket on the token first) or leave that to ``http.client`` on the first request.
    Raises :class:`TimeoutError` when the deadline passes.
    """
    loop = asyncio.get_running_loop()
    done: asyncio.Future[HttpResponse] = loop.create_future()
    cancel = Cancel()

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
        conn: http.client.HTTPConnection | None = None
        response = None
        try:
            if cancel.left:
                return
            conn = open_connection(cancel)
            if not cancel.may_send():
                return
            conn.request(method, target, headers=dict(headers))
            if not cancel.publish(conn.sock):  # cancelled while a lazy connect was running
                return
            response = conn.getresponse()
            length = response.getheader("Content-Length")
            if length is not None and length.isdigit() and int(length) > max_bytes:
                settle(ResponseTooLarge())
                return
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    settle(ResponseTooLarge())
                    return
                chunks.append(chunk)
            names = {k.lower(): v for k, v in response.getheaders()}
            settle(HttpResponse(response.status, names, b"".join(chunks)))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            if not cancel.left:  # otherwise it failed because we stopped it
                settle(exc)
        finally:
            if response is not None:
                response.close()
            if conn is not None:
                conn.close()

    threading.Thread(target=worker, name=thread_name, daemon=True).start()
    try:
        return await asyncio.wait_for(done, timeout)
    except asyncio.TimeoutError:  # not the builtin one before Python 3.11
        raise TimeoutError from None
    finally:
        cancel.leave()
