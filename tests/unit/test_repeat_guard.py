"""A model that keeps sending the same call loses the tool budget, not the turn (issue #58).

``ToolRunner`` already refused an identical call instead of running it, but it only said so:
the turn kept going, so a model that did not read the refusal — observed with Japanese text
behind the window guard (#47), where the same 20KB ``read`` came back thirty times — spent
the whole per-turn budget and the user lost the turn.

The guardrail now gives up on a call after ``max_repeat_refusals`` refusals in a row: that
call gets the terminal instruction the spent budget would have given, and the turn is marked
as wound down. A call that changes its arguments still runs -- that is what the notice asked
for, and on a provider that runs its own tool loop it is the only recovery there is.

These tests drive ``ToolRunner`` directly, because the counting rule is the runner's and
nothing else decides it (principle 2). What a loop does with it is pinned in
``tests/providers/test_chatloop.py``.
"""

from pathlib import Path

from kennel import Agent, MockProvider
from kennel.providers.mock import Text, ToolCall
from kennel.runner import (
    REPEATED_CALL_REFUSAL,
    STOP_CALLING_TOOLS,
    TOOLS_STOPPED_NOTICE,
    WINDOW_EXCEEDED_NOTICE,
    ToolRunner,
)


def runner(tmp_path: Path, **kw) -> ToolRunner:
    """A runner over a workspace holding one small and one over-the-window file."""
    (tmp_path / "small.txt").write_text("alpha\nbeta\ngamma\n")
    # 100 lines of Japanese, ~36KB: over a 4,096-token window whole, small line by line.
    (tmp_path / "ja.txt").write_text(("会議の記録。" * 20 + "\n") * 100)
    agent = Agent(tmp_path, tools=["read", "glob"], provider=MockProvider())
    started = ToolRunner(agent.tools, agent.tool_context(), agent.events, "s", **kw)
    started.begin_turn()
    return started


async def read(run: ToolRunner, path: str = "ja.txt", **arguments) -> str:
    return await run.invoke("read", {"path": path, **arguments})


# -- AC-1 / AC-6: the third refusal in a row ends the turn's tool use ------------------


async def test_the_same_call_is_run_twice_refused_twice_then_ends_the_turn(tmp_path):
    run = runner(tmp_path, context_window_tokens=4096)
    answers = [await read(run) for _ in range(8)]

    # Two results (withheld: the file does not fit the window), two refusals, then the
    # terminal instruction for the rest of the turn — five calls instead of thirty-two.
    assert [a.startswith("Error: the read result is") for a in answers[:2]] == [True, True]
    assert answers[2:4] == [REPEATED_CALL_REFUSAL, REPEATED_CALL_REFUSAL]
    assert answers[4:] == [TOOLS_STOPPED_NOTICE] * 4
    assert [r.status for r in run.records] == ["ok", "ok"] + ["blocked"] * 6
    assert [r.withheld for r in run.records[:2]] == [True, True]  # #47 still measures them


async def test_the_turn_is_stopped_without_the_budget_being_spent(tmp_path):
    run = runner(tmp_path, context_window_tokens=4096)
    for _ in range(5):
        await read(run)

    assert run.repeat_wedged is True
    assert run.limit_hit is False  # the budget was not spent
    assert run.no_more_tool_rounds is True  # what a loop Kennel drives itself asks
    # Giving up on one call is not the turn ending: only a loop that stopped there says so
    # (#66), and until one does the turn is not "tool_limit".
    assert run.tool_rounds_stopped is False
    run.stop_tool_rounds()
    assert run.tool_rounds_stopped is True
    assert run._turn_calls == 5 < run.max_tool_calls


async def test_the_records_tell_a_refusal_apart_from_a_call_given_up_on(tmp_path):
    run = runner(tmp_path, context_window_tokens=4096)
    for _ in range(5):
        await read(run)

    assert run.records[2].error == "repeated identical tool call"
    assert run.records[4].error == "refused 3 times in a row; not run again this turn"


async def test_the_terminal_instruction_is_the_one_a_spent_budget_gives(tmp_path):
    spent = runner(tmp_path, max_tool_calls=1)
    await read(spent, "small.txt")
    budget_answer = await read(spent, "small.txt")

    assert STOP_CALLING_TOOLS in budget_answer and STOP_CALLING_TOOLS in TOOLS_STOPPED_NOTICE
    # Apple's SDK runs its own tool loop and cannot be stopped from a tool callback, so this
    # sentence arriving at the fifth call rather than the thirty-third is the whole lever.
    assert TOOLS_STOPPED_NOTICE.startswith("Error: ")


# -- AC-4: a model that changes its arguments is never in the way of this ---------------


async def test_a_model_that_narrows_its_read_is_not_refused(tmp_path):
    run = runner(tmp_path, context_window_tokens=4096)
    withheld = await read(run)
    narrowed = await read(run, start_line=1, end_line=2)

    assert withheld.startswith("Error: the read result is")
    assert "会議の記録" in narrowed  # the narrowed read was handed over as usual
    assert run.repeat_wedged is False and [r.status for r in run.records] == ["ok", "ok"]


async def test_a_call_that_is_not_a_repeat_resets_the_count(tmp_path):
    run = runner(tmp_path, context_window_tokens=4096)
    answers = []
    for i in range(7):
        answers.append(await read(run) if i % 2 == 0 or i < 3 else await run.invoke("glob", {"pattern": "*.txt"}))

    # Refusals at 3, 5 and 7 with a call that ran in between: never three in a row, so the
    # turn is left alone. A long turn that repeats something twice by accident is not wedged.
    assert run.repeat_wedged is False
    assert [r.status for r in run.records] == ["ok", "ok", "blocked", "ok", "blocked", "ok", "blocked"]
    assert answers[2] == answers[4] == answers[6] == REPEATED_CALL_REFUSAL


async def test_begin_turn_clears_the_wedge(tmp_path):
    run = runner(tmp_path, context_window_tokens=4096)
    for _ in range(5):
        await read(run)
    assert run.repeat_wedged is True

    run.stop_tool_rounds()
    run.begin_turn()
    assert run.repeat_wedged is False and run.no_more_tool_rounds is False
    assert run.tool_rounds_stopped is False
    assert (await read(run)).startswith("Error: the read result is")  # the tool runs again


async def test_a_narrowed_call_still_runs_after_the_guardrail_gave_up(tmp_path):
    """The recovery the notice asks for is never refused, even once the turn wound down.

    Apple's SDK keeps its own tool loop running whatever ``no_more_tool_rounds`` says, so a
    model that finally narrows its read there has to get the lines it asked for -- refusing it
    would turn issue #58's lost turn into a lost answer.
    """
    run = runner(tmp_path, context_window_tokens=4096)
    for _ in range(5):
        await read(run)
    assert run.repeat_wedged is True

    narrowed = await read(run, start_line=1, end_line=2)
    assert "会議の記録" in narrowed and run.records[-1].status == "ok"
    assert await read(run) == TOOLS_STOPPED_NOTICE  # the repeated one stays refused


# -- AC-4: the window notice itself now says not to resend the call --------------------


def test_the_window_notice_tells_the_model_not_to_resend_the_same_call():
    notice = WINDOW_EXCEEDED_NOTICE.format(tool="read", tokens=9002, window=4096)
    assert "start_line/end_line" in notice  # #47: it still says how to ask for less
    assert "Do not send this same call again with the same arguments" in notice


# -- #66 AC-2: a provider that runs its own tool loop keeps answering after the wedge -------


async def test_a_turn_answered_after_the_wedge_by_a_narrowed_call_is_end_turn(tmp_path):
    """Apple's SDK loops by itself and never asks ``no_more_tool_rounds``. The guardrail gives
    up on the repeated read, the model narrows it and answers: the turn did not end at the
    cutoff, so ``stop_reason`` stays ``end_turn`` even though ``repeat_wedged`` is set."""
    (tmp_path / "ja.txt").write_text(("会議の記録。" * 20 + "\n") * 100)
    repeated = [ToolCall("read", {"path": "ja.txt"}) for _ in range(5)]
    narrowed = ToolCall("read", {"path": "ja.txt", "start_line": 1, "end_line": 2})
    provider = MockProvider([[*repeated, narrowed, Text("冒頭は会議の記録です。")]])
    agent = Agent(tmp_path, tools=["read"], provider=provider)
    session = agent.new_session()
    result = await session.run("読んで")

    assert session.runner.repeat_wedged is True  # the mark is there
    assert result.stop_reason == "end_turn" and result.text == "冒頭は会議の記録です。"
    assert [call.status for call in result.tool_calls] == ["ok", "ok", "blocked", "blocked", "blocked", "ok"]
    await session.close()
