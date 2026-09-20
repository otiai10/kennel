"""What a tool result costs, and how the terminal says so by default (issue #32).

The on-device window is 4,096 tokens, so a single 20KB file cannot fit it. The runner counts
the bytes, so the runner is where the estimate and the comparison live; the renderer only
draws them, without needing --verbose.
"""

import io

from kennel import Agent, EventType, KennelConfig, MockProvider
from kennel.cli.renderer import Renderer
from kennel.events import Event
from kennel.providers.mock import Text, ToolCall

BIG = "x" * 20_000


def big_workspace(tmp_path):
    (tmp_path / "big.txt").write_text(BIG)
    (tmp_path / "small.txt").write_text("hello\n")
    return tmp_path


def render(data, **renderer_kw) -> str:
    out = io.StringIO()
    renderer = Renderer(out=out, err=io.StringIO(), color=False, **renderer_kw)
    renderer.on_event(Event(EventType.TOOL_COMPLETED, "sid", data))
    return out.getvalue()


# -- the event ----------------------------------------------------------------


async def test_a_result_too_big_for_the_window_says_so_on_the_event(tmp_path):
    """AC-1: estimated_tokens and window_exceeded=True for 20,000 bytes in a 4,096 window."""
    ws = big_workspace(tmp_path)
    turns = [[ToolCall("read", {"path": "big.txt"}), ToolCall("read", {"path": "small.txt"}), Text("done")]]
    agent = Agent(ws, tools=["read"], provider=MockProvider(turns, context_window_tokens=4096))
    events: list[Event] = []
    agent.events.subscribe(events.append)
    await agent.run("read them")
    completed = [e.data for e in events if e.type == EventType.TOOL_COMPLETED]
    assert len(completed) == 2
    big, small = completed
    assert big["output_bytes"] > 20_000
    assert big["estimated_tokens"] == -(-big["output_bytes"] // 4)  # ceil(bytes / 4)
    assert big["context_window_tokens"] == 4096 and big["window_exceeded"] is True
    assert small["estimated_tokens"] < 4096 and small["window_exceeded"] is False


async def test_without_a_declared_window_the_question_is_not_answered(tmp_path):
    """A provider that declares no window cannot make the comparison, so it says no window."""
    ws = big_workspace(tmp_path)
    turns = [[ToolCall("read", {"path": "big.txt"}), Text("done")]]
    agent = Agent(ws, tools=["read"], provider=MockProvider(turns, context_window_tokens=None))
    events: list[Event] = []
    agent.events.subscribe(events.append)
    await agent.run("read it")
    data = next(e.data for e in events if e.type == EventType.TOOL_COMPLETED)
    assert data["estimated_tokens"] > 4096  # still reported
    assert data["context_window_tokens"] is None and data["window_exceeded"] is False


# -- the line the terminal draws ----------------------------------------------


def test_the_size_line_is_drawn_without_verbose():
    """AC-2: bytes and "exceeds the" appear with Renderer(verbose=False)."""
    line = render(
        {"tool": "read", "summary": "Read big.txt", "output_bytes": 20_008, "estimated_tokens": 5002,
         "context_window_tokens": 4096, "window_exceeded": True, "truncated": False, "duration_ms": 1.2},
        verbose=False,
    )
    assert "bytes" in line and "exceeds the" in line
    assert line.strip() == "↳ 20,008 bytes (~5,002 tokens, exceeds the 4,096-token window)"
    assert "ms" not in line  # the timing is what --verbose adds


def test_a_result_that_fits_is_reported_plainly_and_verbose_adds_the_timing():
    data = {"tool": "read", "summary": "Read small.txt", "output_bytes": 13, "estimated_tokens": 4,
            "context_window_tokens": 4096, "window_exceeded": False, "truncated": True, "duration_ms": 3.0}
    assert render(data, verbose=False).strip() == "↳ 13 bytes (~4 tokens, truncated)"
    assert render(data, verbose=True).strip() == "↳ 13 bytes (~4 tokens, truncated) in 3 ms"


def test_the_line_survives_an_event_without_the_new_keys():
    """A consumer emitting a bare tool.completed must not break the renderer."""
    assert render({"tool": "read", "output_bytes": 12}, verbose=False).strip() == "↳ 12 bytes"


def test_quiet_draws_nothing():
    assert render({"tool": "read", "output_bytes": 12}, verbose=False, quiet=True) == ""


def test_the_warning_is_yellow_only_when_the_window_is_exceeded():
    exceeded = {"output_bytes": 20_008, "estimated_tokens": 5002, "context_window_tokens": 4096, "window_exceeded": True}
    out = io.StringIO()
    Renderer(out=out, err=io.StringIO(), color=True).on_event(Event(EventType.TOOL_COMPLETED, "s", exceeded))
    assert "\x1b[33m" in out.getvalue()  # yellow
    fits = {**exceeded, "window_exceeded": False}
    out = io.StringIO()
    Renderer(out=out, err=io.StringIO(), color=True).on_event(Event(EventType.TOOL_COMPLETED, "s", fits))
    assert "\x1b[2m" in out.getvalue()  # dim


# -- the per-turn context line ------------------------------------------------


def test_note_is_written_by_default_and_suppressed_when_quiet():
    """AC-3, the renderer's half: the context line is a note, and notes obey --quiet."""
    out = io.StringIO()
    Renderer(out=out, err=io.StringIO(), color=False, verbose=False).note("(context: 12% of 4096 tokens)")
    assert out.getvalue().startswith("(context")
    out = io.StringIO()
    Renderer(out=out, err=io.StringIO(), color=False, verbose=False, quiet=True).note("(context: ...)")
    assert out.getvalue() == ""


async def test_the_context_summary_still_says_it_is_an_estimate(tmp_path):
    """Principle 4: the number the line shows must never look measured."""
    agent = Agent(big_workspace(tmp_path), tools=["read"], provider=MockProvider(["hi"]), config=KennelConfig())
    session = agent.new_session()
    await session.run("q")
    summary = session.context_usage().summary()
    assert "estimated" in summary and "of 4096 tokens" in summary
