"""Event bus used by the agent to report progress to SDK consumers.

The CLI renderer is one consumer; applications can subscribe their own. Events
carry summaries and metadata, never raw file contents or secrets by default.
Emission may happen from a worker thread (provider tool callbacks), so the bus
is thread-safe and subscriber exceptions are logged rather than propagated.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
