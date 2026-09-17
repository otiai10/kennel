from kennel.errors import (
    ContextLimitError,
    KennelError,
    ModelUnavailableError,
    ProviderError,
    ToolArgumentError,
    ToolError,
    WorkspaceError,
    WorkspaceEscapeError,
)
from kennel.events import EventBus


def test_hierarchy():
    assert issubclass(ModelUnavailableError, ProviderError) and issubclass(ProviderError, KennelError)
    assert issubclass(ContextLimitError, ProviderError)
    assert issubclass(ToolArgumentError, ToolError)
    assert issubclass(WorkspaceEscapeError, WorkspaceError)
    assert str(ToolArgumentError("bad arg")) == "bad arg"


def test_bus_subscribe_emit_unsubscribe():
    bus = EventBus()
    seen = []
    unsubscribe = bus.subscribe(seen.append)
    bus.subscribe(lambda e: 1 / 0)  # a broken subscriber must not break emission
    ev = bus.emit("tool.started", "s1", tool="glob")
    assert ev.type == "tool.started" and ev.session_id == "s1" and ev.data == {"tool": "glob"}
    assert seen == [ev] and ev.timestamp > 0
    unsubscribe()
    bus.emit("x", "s1")
    assert len(seen) == 1
