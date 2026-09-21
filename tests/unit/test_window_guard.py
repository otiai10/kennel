"""A tool result bigger than the whole context window is withheld (issue #47).

The on-device window is 4,096 tokens and the output limit is 64KB, so a 20KB file passes the
limit and then cannot fit the window. Handing it to the model does not shorten the
conversation — the request fails (Apple: status 255) — so ``ToolRunner.invoke`` keeps it out
and gives the model a one-line instruction to ask for a smaller part instead.

The guard changes no limit: ``tools.max_output_bytes`` is still what bounds every result.
"""

import io
import json

from kennel import Agent, EventType, MockProvider
from kennel.cli.renderer import Renderer
from kennel.events import Event
from kennel.providers.mock import Text, ToolCall
from kennel.runner import WINDOW_EXCEEDED_NOTICE


def workspace(tmp_path, **config):
    (tmp_path / "big.txt").write_text("x" * 20_000)
    (tmp_path / "small.txt").write_text("hello\n")
    if config:
        (tmp_path / "kennel.json").write_text(json.dumps(config))
    return tmp_path


def build(tmp_path, *, reads=("big.txt",), window=4096, **config):
    """An agent scripted to read ``reads`` and answer, with the events it emitted."""
    ws = workspace(tmp_path, **config)
    turns = [[*(ToolCall("read", {"path": p}) for p in reads), Text("done")]]
    provider = MockProvider(turns, context_window_tokens=window)
    agent = Agent(ws, tools=["read"], provider=provider)
    events: list[Event] = []
    agent.events.subscribe(events.append)
    return agent, provider, events


def completed(events):
    return [e.data for e in events if e.type == EventType.TOOL_COMPLETED]


# -- AC-1: the result is measured, withheld, and replaced by an instruction ----


async def test_a_result_too_big_for_the_window_is_not_handed_to_the_model(tmp_path):
    agent, provider, events = build(tmp_path)
    result = await agent.run("read it")

    handed = provider.sessions[0].tool_results[0]
    assert "xxxx" not in handed  # the file's content never reached the model
    assert handed.startswith("Error: the read result is ~5,002 tokens")
    assert "4,096-token context window" in handed
    assert "start_line/end_line" in handed  # it says how to ask for less

    data = completed(events)[0]
    assert data["output_bytes"] > 20_000  # the measurement is of the result, not of the notice
    assert data["estimated_tokens"] == -(-data["output_bytes"] // 4)
    assert data["window_exceeded"] is True and data["withheld"] is True
    assert data["truncated"] is False  # 20KB is under the 64KB output limit

    record = result.tool_calls[0]
    assert record.status == "ok" and record.withheld is True  # the tool itself succeeded
    assert record.output_bytes == data["output_bytes"]


def test_the_notice_is_short_enough_to_be_worth_sending():
    """Replacing a 5,000-token result with something comparable would solve nothing."""
    notice = WINDOW_EXCEEDED_NOTICE.format(tool="read", tokens=5002, window=4096)
    assert len(notice.encode()) < 400


async def test_a_result_that_fits_is_handed_over_unchanged(tmp_path):
    agent, provider, events = build(tmp_path, reads=("small.txt",))
    await agent.run("read it")
    assert "hello" in provider.sessions[0].tool_results[0]
    assert completed(events)[0]["withheld"] is False


async def test_a_provider_without_a_declared_window_withholds_nothing(tmp_path):
    """No window, no comparison: the question cannot be answered, so nothing is kept back."""
    agent, provider, events = build(tmp_path, window=None)
    await agent.run("read it")
    assert "xxxx" in provider.sessions[0].tool_results[0]
    data = completed(events)[0]
    assert data["window_exceeded"] is False and data["withheld"] is False


async def test_the_model_can_come_back_with_a_range(tmp_path):
    """The instruction is actionable: a narrowed read goes through as usual."""
    # 400 lines (the read limit) of 60 characters still make ~6,000 tokens.
    (tmp_path / "long.txt").write_text("".join(f"line {i} " + "y" * 60 + "\n" for i in range(1, 3001)))
    turns = [[
        ToolCall("read", {"path": "long.txt"}),
        ToolCall("read", {"path": "long.txt", "start_line": 1, "end_line": 20}),
        Text("done"),
    ]]
    provider = MockProvider(turns, context_window_tokens=4096)
    agent = Agent(tmp_path, tools=["read"], provider=provider)
    result = await agent.run("read it")
    first, second = provider.sessions[0].tool_results
    assert first.startswith("Error: the read result is")
    assert "line 20" in second
    assert [c.withheld for c in result.tool_calls] == [True, False]


# -- AC-2: the output limit stays the only limit ------------------------------


async def test_an_explicit_output_limit_is_what_bounds_the_result(tmp_path):
    """``tools.max_output_bytes`` below the window brings truncated content back."""
    agent, provider, events = build(tmp_path, tools={"max_output_bytes": 8192})
    assert agent.config.max_tool_output_bytes == 8192  # the project config decides
    await agent.run("read it")

    data = completed(events)[0]
    assert data["output_bytes"] == 8192 and data["truncated"] is True
    assert data["estimated_tokens"] == 2048 and data["window_exceeded"] is False
    assert data["withheld"] is False
    handed = provider.sessions[0].tool_results[0]
    assert "xxxx" in handed and handed.endswith("[output truncated]")


async def test_the_window_never_lowers_an_explicit_output_limit(tmp_path):
    """The guard derives no limit from the window: a large explicit value is not clamped."""
    agent, _, events = build(tmp_path, tools={"max_output_bytes": 100_000})
    await agent.run("read it")
    data = completed(events)[0]
    assert data["output_bytes"] > 20_000 and data["truncated"] is False
    assert data["withheld"] is True  # measured in full, then withheld — not truncated to fit


# -- the estimate and the line the terminal draws -----------------------------


async def test_a_withheld_result_is_not_counted_against_the_window(tmp_path):
    """It never reached the provider session, so the estimate must not pretend it did."""
    ws = workspace(tmp_path)
    turns = [[ToolCall("read", {"path": "big.txt"}), Text("done")]]
    agent = Agent(ws, tools=["read"], provider=MockProvider(turns, context_window_tokens=4096))
    session = agent.new_session()
    await session.run("read it")
    usage = session.context_usage()
    assert usage.estimated is True and usage.used_tokens < 4096  # not the 5,002 it would have cost


def test_the_size_line_says_the_result_was_withheld():
    out = io.StringIO()
    Renderer(out=out, err=io.StringIO(), color=False).on_event(
        Event(EventType.TOOL_COMPLETED, "sid", {
            "tool": "read", "output_bytes": 20_008, "estimated_tokens": 5002,
            "context_window_tokens": 4096, "window_exceeded": True, "withheld": True,
        })
    )
    assert out.getvalue().strip() == "↳ 20,008 bytes (~5,002 tokens, exceeds the 4,096-token window, withheld)"
