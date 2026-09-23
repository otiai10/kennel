"""Saved conversations: what ``kennel --continue`` / ``--resume <id>`` read back (#62).

A transcript is the *restorable* record of a session and therefore carries text — the
prompts and the answers. That is the difference from the event log (:mod:`kennel.events`),
which carries sizes and summaries only and may be on by default for that reason. A
transcript may not, so saving one is opt-in: an :class:`~kennel.Agent` saves only when it is
given a :class:`SessionStore` (the CLI does that for ``--persist`` / ``sessions.persist``).

What is kept per turn is the prompt, the answer, the stop reason and, for each tool call,
its name and outcome. Tool arguments, summaries and output are not kept: ``shell``'s summary
is the whole command line and ``write``'s arguments are the file body, and restoring a
conversation only needs the prompts and answers (the summary the next provider session is
seeded with is built from those) plus whether a turn called tools at all (the nudge check).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import ConfigurationError, SessionError
from .events import state_dir as default_state_dir
from .runner import ToolCallRecord
from .session import Turn
from .workspace import Workspace

log = logging.getLogger(__name__)

#: The transcript line format. A line with any other ``v`` is refused rather than guessed at.
TRANSCRIPT_VERSION = 1

_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def check_session_id(session_id: str) -> str:
    """Return ``session_id`` if it may name a transcript, else raise :class:`ConfigurationError`.

    Only ``[A-Za-z0-9_-]{1,64}``: an id ends up in a file name, so nothing that could walk out
    of the transcript directory (``..``, ``/``) gets that far.
    """
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ConfigurationError(
            f"invalid session id {session_id!r}: use 1-64 letters, digits, '-' or '_'"
        )
    return session_id


@runtime_checkable
class SessionStore(Protocol):
    """Where a :class:`~kennel.Session` saves its turns and reads them back.

    ``append`` is called once per turn, right after the turn lands in the history (failed
    turns included); an exception from it is logged and does not fail the turn. ``load``
    returns ``[]`` for an id with nothing saved. ``latest`` is the id of the most recently
    updated conversation, or ``None``. ``clear`` forgets a conversation and must raise if it
    could not. ``lock`` claims an id for as long as a session has it open and returns the
    function that releases it; it raises :class:`~kennel.errors.ConfigurationError` when the id
    is in use elsewhere.
    """

    def append(self, session_id: str, turn: Turn) -> None: ...

    def load(self, session_id: str) -> list[Turn]: ...

    def latest(self) -> str | None: ...

    def clear(self, session_id: str) -> None: ...

    def lock(self, session_id: str) -> Callable[[], None]: ...


class FileSessionStore:
    """Transcripts as JSON Lines files, one per session, kept per workspace.

    Files live in ``<state_dir>/transcripts/<workspace-hash>/<session_id>.jsonl``, where the
    hash is the first 16 hex digits of the sha256 of the workspace's resolved path, so
    :meth:`latest` and :meth:`load` only ever see the conversations of this workspace. The
    directory is kept apart from the event log's ``sessions/`` on purpose: one holds text and
    is restored from, the other holds sizes only and is looked at. Directories are ``0700``
    and files ``0600`` whatever the umask or an existing directory says.

    ``persist=False`` makes a store that reads (and clears) but never appends: the CLI uses it
    to resume a conversation while saving is off, so the resumed session is not written to.

    Example::

        agent = Agent("~/meetings", session_store=FileSessionStore("~/meetings"))
        session = agent.new_session()
        ...
        later = agent.new_session(session_id=session.id)   # history restored
    """

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        state_dir: str | os.PathLike[str] | None = None,
        persist: bool = True,
    ) -> None:
        root = Workspace(workspace_root).root
        digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        base = Path(state_dir).expanduser() if state_dir is not None else default_state_dir()
        self.directory = base / "transcripts" / digest
        self.persist = persist

    # -- paths ----------------------------------------------------------------

    def path_for(self, session_id: str) -> Path:
        """The transcript file for ``session_id`` (validated; see :func:`check_session_id`)."""
        path = self.directory / f"{check_session_id(session_id)}.jsonl"
        if path.parent != self.directory:  # belt and braces: the pattern already rules this out
            raise ConfigurationError(f"invalid session id {session_id!r}")
        return path

    def _ensure_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for directory in (self.directory.parent, self.directory):
            os.chmod(directory, 0o700)

    def _open_private(self, path: Path, flags: int) -> int:
        self._ensure_directory()
        fd = os.open(path, flags | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)  # an existing file, or a umask that stripped nothing
        except OSError:
            os.close(fd)
            raise
        return fd

    # -- SessionStore ---------------------------------------------------------

    def append(self, session_id: str, turn: Turn) -> None:
        if not self.persist:
            return
        line = json.dumps(_record(turn), ensure_ascii=False) + "\n"
        fd = self._open_private(self.path_for(session_id), os.O_RDWR)
        try:
            _drop_partial_line(fd)
            os.lseek(fd, 0, os.SEEK_END)
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    def load(self, session_id: str) -> list[Turn]:
        path = self.path_for(session_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeDecodeError) as exc:
            raise SessionError(f"cannot read the saved conversation {path}: {exc}") from exc
        lines = text.split("\n")
        complete = lines[-1] == ""  # the file ends with a newline, so nothing is cut short
        if complete:
            lines.pop()
        turns: list[Turn] = []
        for number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if number == len(lines) and not complete:
                    # A crash in the middle of a write: that turn is lost, the rest is good.
                    log.warning("%s: ignoring the last line, which was cut short", path)
                    break
                raise SessionError(f"{path}:{number}: the saved conversation is corrupt") from None
            turns.append(_turn(record, f"{path}:{number}"))
        return turns

    def latest(self) -> str | None:
        try:
            candidates = [p for p in self.directory.glob("*.jsonl") if _SESSION_ID.fullmatch(p.stem)]
        except OSError:
            return None
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime_ns).stem

    def clear(self, session_id: str) -> None:
        path = self.path_for(session_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise SessionError(f"cannot delete the saved conversation {path}: {exc}") from exc

    def lock(self, session_id: str) -> Callable[[], None]:
        """Claim ``session_id`` with a non-blocking ``flock`` on ``<session_id>.lock``.

        Raises :class:`ConfigurationError` when another session (in this process or another)
        holds it. When the lock file cannot be created at all — an unwritable state directory —
        nothing could be saved there either, so this warns and carries on unlocked rather than
        refusing to open the session.
        """
        lock_path = self.path_for(session_id).with_suffix(".lock")
        while True:
            try:
                fd = self._open_private(lock_path, os.O_RDWR)
            except OSError as exc:
                log.warning("session %s is not locked: cannot create %s: %s", session_id, lock_path, exc)
                return _no_op
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                raise ConfigurationError(f"session {session_id} is in use (open in another session or process)") from None
            except OSError:
                os.close(fd)
                raise
            # The holder unlinks the file when it lets go; if that happened between our open
            # and our flock, we hold a lock on a file nobody else can see. Try again.
            try:
                same = os.fstat(fd).st_ino == os.stat(lock_path).st_ino
            except FileNotFoundError:
                same = False
            if same:
                break
            os.close(fd)

        def release() -> None:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            os.close(fd)  # closing drops the flock

        return release


def _no_op() -> None:
    return None


def _drop_partial_line(fd: int) -> None:
    """Cut a line a crash left unfinished, so the next line does not get glued onto it."""
    size = os.lseek(fd, 0, os.SEEK_END)
    if size == 0 or os.pread(fd, 1, size - 1) == b"\n":
        return
    data = os.pread(fd, size, 0)
    os.ftruncate(fd, data.rfind(b"\n") + 1)


def _record(turn: Turn) -> dict[str, Any]:
    return {
        "v": TRANSCRIPT_VERSION,
        "t": round(time.time(), 3),
        "prompt": turn.prompt,
        "response": turn.response,
        "stop_reason": turn.stop_reason,
        "tools": [{"name": call.name, "status": call.status} for call in turn.tool_calls],
    }


def _turn(record: Any, where: str) -> Turn:
    if not isinstance(record, dict):
        raise SessionError(f"{where}: the saved conversation is corrupt")
    version = record.get("v")
    if version != TRANSCRIPT_VERSION:
        raise SessionError(f"{where}: unsupported transcript version {version!r} (this Kennel reads {TRANSCRIPT_VERSION})")
    try:
        prompt, response, stop_reason = record["prompt"], record["response"], record["stop_reason"]
        tools = [(tool["name"], tool["status"]) for tool in record["tools"]]
    except (KeyError, TypeError):
        raise SessionError(f"{where}: the saved conversation is corrupt") from None
    # Name and outcome only: the rest was never saved (see the module docstring).
    calls = [ToolCallRecord(name, {}, name, status) for name, status in tools]
    return Turn(prompt, response, stop_reason, calls)
