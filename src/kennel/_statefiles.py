"""Files Kennel keeps about a session: the one rule for session ids and the one way to open them.

Both the event log (:mod:`kennel.events`) and saved conversations (:mod:`kennel.sessions`) name
a file after a session id and keep it private, so both rules live here once (constitution
principle 2): an id that could walk out of its directory is refused before any path is built,
and every such file is ``0600`` in ``0700`` directories whatever the umask says (#83).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

from .errors import ConfigurationError

#: What a session id may look like: it ends up in a file name.
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


def check_session_id(session_id: str) -> str:
    """Return ``session_id`` if it may name a session's files, else raise :class:`ConfigurationError`.

    Only ``[A-Za-z0-9_-]{1,64}``: an id ends up in a file name, so nothing that could walk out
    of the directory (``..``, ``/``) gets that far.
    """
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ConfigurationError(
            f"invalid session id {session_id!r}: use 1-64 letters, digits, '-' or '_'"
        )
    return session_id


def open_private(path: Path, flags: int, *, directories: Iterable[Path] = ()) -> int:
    """Open ``path`` (creating it) as a ``0600`` file and return the descriptor.

    Parent directories that do not exist yet are created ``0700``; ``directories`` names
    existing ones the caller owns and wants tightened to ``0700`` as well. The file is ``0600``
    even when it already existed or the umask would have left more. Raises :class:`OSError`.
    """
    missing: list[Path] = []
    parent = path.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)  # mkdir's mode is masked by the umask
    for directory in directories:
        os.chmod(directory, 0o700)
    fd = os.open(path, flags | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, 0o600)  # an existing file, or a umask that stripped nothing
    except OSError:
        os.close(fd)
        raise
    return fd
