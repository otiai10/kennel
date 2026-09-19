"""Event bus used by the agent to report progress to SDK consumers.

The CLI renderer is one consumer; applications can subscribe their own. Events
carry summaries and metadata, never raw file contents or secrets by default.
Emission may happen from a worker thread (provider tool callbacks), so the bus
is thread-safe and subscriber exceptions are logged rather than propagated.

:class:`JsonlEventLog` is the subscriber that keeps a session's events on disk, in
the same flat JSON form ``--trace`` writes to stderr (:func:`event_json`). It writes
to a local file and nothing else: no network, no backend (constitution principle 1).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)


class EventType:
    SESSION_STARTED = "session.started"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    SESSION_CANCELLED = "session.cancelled"
    MODEL_STARTED = "model.started"
    MODEL_DELTA = "model.delta"
    MODEL_COMPLETED = "model.completed"
    MODEL_NUDGED = "model.nudged"
    TOOL_REQUESTED = "tool.requested"
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_DENIED = "permission.denied"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    TOOL_BLOCKED = "tool.blocked"
    CONTEXT_COMPACTED = "context.compacted"


@dataclass(frozen=True)
class Event:
    type: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register ``callback`` and return a function that unsubscribes it."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def emit(self, type: str, session_id: str, **data: Any) -> Event:
        event = Event(type=type, session_id=session_id, data=data)
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub(event)
            except Exception:  # noqa: BLE001 - a broken consumer must not break the agent
                log.exception("event subscriber failed for %s", type)
        return event


def event_json(event: Event) -> str:
    """One JSON line for ``event``: ``t``, ``type``, ``session_id`` and the event's own data.

    Shared by ``--trace`` (stderr) and :class:`JsonlEventLog` (a file) so the two forms
    cannot drift. It carries exactly what the bus carries — summaries, sizes and timings,
    never file contents or generated text (constitution principle 3), which is what makes
    keeping the log on by default safe.
    """
    record = {"t": round(event.timestamp, 3), "type": event.type, "session_id": event.session_id, **event.data}
    return json.dumps(record, ensure_ascii=False, default=str)


def state_dir() -> Path:
    """Where Kennel keeps local state, resolved at call time.

    ``KENNEL_STATE_DIR`` wins, then ``XDG_STATE_HOME/kennel``, then
    ``~/.local/state/kennel``. Nothing here leaves the machine.
    """
    override = os.environ.get("KENNEL_STATE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return base / "kennel"


def session_log_path(session_id: str) -> Path:
    """The event log file for one session: ``<state_dir>/sessions/<session_id>.jsonl``."""
    return state_dir() / "sessions" / f"{session_id}.jsonl"


class JsonlEventLog:
    """An :class:`EventBus` subscriber that appends events to a JSON Lines file.

    One line per event in the form of :func:`event_json`. Pass ``session_id`` to keep one
    session's events out of another's file when several sessions share a bus.

    Observation must stay silent: a file that cannot be opened or written is reported once
    through ``logging`` and the log then does nothing, so a full disk or a read-only
    directory never stops the agent (constitution principle 3). Usable as a context manager
    and safe to call from any thread.

    Example::

        with JsonlEventLog("/tmp/kennel.jsonl") as log:
            unsubscribe = agent.events.subscribe(log)
    """

    def __init__(self, path: str | os.PathLike[str], *, session_id: str | None = None) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self._lock = threading.Lock()
        self._file: IO[str] | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._file = self.path.open("a", encoding="utf-8")
        except OSError as exc:
            log.warning("event log disabled: cannot open %s: %s", self.path, exc)

    @property
    def enabled(self) -> bool:
        """False once the file could not be opened, or a write failed and gave up."""
        return self._file is not None

    def __call__(self, event: Event) -> None:
        if self.session_id is not None and event.session_id != self.session_id:
            return
        with self._lock:
            file = self._file
            if file is None:
                return
            try:
                file.write(event_json(event) + "\n")
                file.flush()  # the point of the log is to survive the crash that follows
            except OSError as exc:
                log.warning("event log disabled: cannot write %s: %s", self.path, exc)
                self._file = None
                self._shut(file)

    def close(self) -> None:
        with self._lock:
            file, self._file = self._file, None
            self._shut(file)

    @staticmethod
    def _shut(file: IO[str] | None) -> None:
        if file is not None:
            try:
                file.close()
            except OSError:  # pragma: no cover - closing a broken file
                pass

    def __enter__(self) -> JsonlEventLog:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
