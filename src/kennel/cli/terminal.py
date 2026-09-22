"""Terminal input plumbing shared by the REPL and the permission prompt."""

from __future__ import annotations

import sys
from typing import IO

try:  # POSIX only; on other platforms there is no tty queue to drop.
    import termios
except ImportError:  # pragma: no cover - non-POSIX
    termios = None  # type: ignore[assignment]


def discard_typed_ahead(stream: IO[str] | None = None) -> None:
    """Drop whatever was typed into *stream* before this prompt was printed.

    The terminal echoes keystrokes as they are typed, so anything entered while the agent
    was busy is already on screen and already queued in the tty. Dropping that queue keeps
    it from being read as the next turn, or as the answer to a permission question nobody
    saw. The echoed characters stay on screen: the terminal mode is never touched, so a
    hard crash cannot leave the user with a terminal that does not echo.

    A no-op when the stream is not a tty (piped input, ``--output-format json``) and when
    the platform has no ``termios``. Never raises: this is plumbing, not policy, and it
    must not fail a turn.
    """
    if termios is None:  # pragma: no cover - non-POSIX
        return
    stream = sys.stdin if stream is None else stream
    try:
        if not stream.isatty():
            return
        termios.tcflush(stream.fileno(), termios.TCIFLUSH)
    except (termios.error, OSError, ValueError):
        return  # a closed, detached or exotic stdin is not worth failing a turn over
