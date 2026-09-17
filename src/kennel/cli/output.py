"""Machine-readable stdout for ``kennel -p``.

``--output-format json`` prints one result object when the turn ends; ``stream-json``
prints one JSON object per line as the turn unfolds and the same result object last;
``--json-schema`` with text output prints the structured document alone. The schema is
documented in ``docs/output-format.md``.

Text deltas are written from the CLI's ``on_delta`` callback rather than from the
``model.delta`` event, because events deliberately carry sizes and summaries
only, never generated or file content (see ``kennel.events``).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import IO, Any

from ..events import Event, EventBus, EventType
from ..session import AgentResult

FORMATS = ("text", "json", "stream-json")


class MachineOutput(ABC):
    """A stdout that belongs to a machine: the renderer stays quiet, this writes.

    Subclasses decide what lands there; ``main`` picks one and never writes to
    stdout itself.
    """

    def attach(self, events: EventBus) -> None:
        return None

    def delta(self, text: str) -> None:
        return None

    @abstractmethod
    def result(self, result: AgentResult) -> None: ...

    def error(self, message: str, *, stop_reason: str = "error") -> None:
        return None  # the message is on stderr already


class DocumentOutput(MachineOutput):
    """``--json-schema`` with text output: stdout is the structured JSON document."""

    def __init__(self, out: IO[str] = sys.stdout) -> None:
        self.out = out

    def result(self, result: AgentResult) -> None:
        if result.text:
            self.out.write(result.text.rstrip("\n") + "\n")
            self.out.flush()


class JsonOutput(MachineOutput):
    """Writes the result (and, for ``stream-json``, every event) to stdout."""

    def __init__(self, format: str, out: IO[str] = sys.stdout) -> None:
        self.format = format
        self.out = out
        self.streaming = format == "stream-json"
        self._lock = threading.Lock()
        self._session_id = ""

    def attach(self, events: EventBus) -> None:
        if self.streaming:
            events.subscribe(self.on_event)

    # -- event consumer -------------------------------------------------------

    def on_event(self, event: Event) -> None:
        self._session_id = event.session_id
        if event.type == EventType.MODEL_DELTA:
            return  # written by delta() instead, with the text itself
        self._write({"type": event.type, "session_id": event.session_id, "timestamp": round(event.timestamp, 3), "data": event.data})

    def delta(self, text: str) -> None:
        if self.streaming and text:
            self._write({"type": EventType.MODEL_DELTA, "session_id": self._session_id, "timestamp": round(time.time(), 3), "data": {"text": text}})

    # -- result ---------------------------------------------------------------

    def result(self, result: AgentResult) -> None:
        self._write(result.to_dict())

    def error(self, message: str, *, stop_reason: str = "error") -> None:
        """Report a failure as a result record; the exit code is unchanged."""
        self._write(
            AgentResult(
                text="",
                stop_reason=stop_reason,
                tool_calls=[],
                usage=None,
                session_id=self._session_id,
                is_error=True,
                error=message,
            ).to_dict()
        )

    def _write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.out.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self.out.flush()
