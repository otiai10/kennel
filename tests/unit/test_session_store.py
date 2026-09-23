"""Saved conversations (#62): what a transcript keeps, where it lives, and how resuming behaves.

Every test runs against the ``KENNEL_STATE_DIR`` the autouse ``state_dir`` fixture points at a
tmp directory, so nothing reaches the real home directory.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest

from kennel import (
    Agent,
    Approval,
    ConfigurationError,
    FileSessionStore,
    KennelConfig,
    MockProvider,
    SessionError,
    SessionStore,
)
from kennel.config import apply_config
from kennel.providers.mock import Text, ToolCall

TOOL_TURN = [ToolCall("read", {"path": "transcripts/2026-09-16.txt"}), Text("It ships on Friday.")]


def make_agent(ws, turns, store=None, **kw) -> Agent:
    return Agent(ws, provider=MockProvider(list(turns)), session_store=store, **kw)


def lines_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def save_two_turns(ws, store) -> str:
    session = make_agent(ws, [TOOL_TURN, "Bob owns the README."], store).new_session()
    await session.run("When do we ship?")
    await session.run("Who owns the README?")
    await session.close()
    return session.id


def test_the_file_store_satisfies_the_protocol(meeting_ws):
    assert isinstance(FileSessionStore(meeting_ws), SessionStore)


# -- AC-1: turns come back ----------------------------------------------------


async def test_a_resumed_session_gets_its_turns_back(meeting_ws):
    store = FileSessionStore(meeting_ws)
    session_id = await save_two_turns(meeting_ws, store)

    resumed = make_agent(meeting_ws, [], store).new_session(session_id=session_id)
    assert resumed.id == session_id
    assert [(t.prompt, t.response, t.stop_reason) for t in resumed.history] == [
        ("When do we ship?", "It ships on Friday.", "end_turn"),
        ("Who owns the README?", "Bob owns the README.", "end_turn"),
    ]
    (call,) = resumed.history[0].tool_calls
    assert (call.name, call.status, call.arguments, call.summary) == ("read", "ok", {}, "read")
    assert resumed.history[1].tool_calls == []
    await resumed.close()


async def test_a_failed_turn_is_saved_too(meeting_ws):
    from kennel.errors import ProviderError
    from kennel.providers.mock import Raise

    store = FileSessionStore(meeting_ws)
    session = make_agent(meeting_ws, [[Raise(ProviderError("boom"))]], store).new_session()
    with pytest.raises(ProviderError):
        await session.run("hello")
    await session.close()
    (line,) = lines_of(store.path_for(session.id))
    assert (line["prompt"], line["stop_reason"]) == ("hello", "error")


# -- AC-2: the first resumed turn starts from the summary ---------------------


async def test_the_first_resumed_turn_is_seeded_with_the_summary(meeting_ws):
    store = FileSessionStore(meeting_ws)
    session_id = await save_two_turns(meeting_ws, store)

    agent = make_agent(meeting_ws, ["Friday, as agreed."], store)
    session = agent.new_session(session_id=session_id)
    assert session.compactions == 0 and session.status()["resumed"] is True
    await session.run("Remind me of the date?")
    instructions = agent.provider.sessions[0].instructions
    assert "Summary of the conversation so far" in instructions
    assert "User asked: When do we ship?" in instructions
    assert "Assistant answered: Bob owns the README." in instructions
    assert session.compactions == 0  # seeding from the saved history is not a compaction
    assert session.last_result.compactions == 0
    await session.close()


async def test_a_new_session_is_not_resumed(meeting_ws):
    session = make_agent(meeting_ws, [], FileSessionStore(meeting_ws)).new_session()
    assert session.resumed is False and session.status()["resumed"] is False
    await session.close()


# -- AC-3: location and permissions ------------------------------------------


async def test_transcripts_live_apart_from_the_event_log(meeting_ws, state_dir):
    store = FileSessionStore(meeting_ws)
    session = make_agent(meeting_ws, ["ok"], store, config=KennelConfig(log_events=True)).new_session()
    await session.run("hi")
    await session.close()
    digest = hashlib.sha256(str(meeting_ws.resolve()).encode()).hexdigest()[:16]
    transcript = state_dir / "transcripts" / digest / f"{session.id}.jsonl"
    assert store.path_for(session.id) == transcript
    assert transcript.is_file() and session.log_path.is_file()
    assert session.log_path == state_dir / "sessions" / f"{session.id}.jsonl"
    assert transcript != session.log_path
    assert lines_of(transcript)[0]["prompt"] == "hi"  # the two files hold different things
    assert all("prompt" not in line for line in lines_of(session.log_path))
    assert not list(transcript.parent.glob("*.lock"))  # the lock goes with the session


async def test_owner_only_permissions_whatever_the_umask(meeting_ws, state_dir):
    (state_dir / "transcripts").mkdir(parents=True, mode=0o755)
    os.chmod(state_dir / "transcripts", 0o755)  # an existing, looser directory
    store = FileSessionStore(meeting_ws)
    previous = os.umask(0)
    try:
        session = make_agent(meeting_ws, ["ok"], store).new_session()
        await session.run("hi")
        await session.close()
    finally:
        os.umask(previous)
    transcript = store.path_for(session.id)
    assert os.stat(transcript).st_mode & 0o777 == 0o600
    assert os.stat(transcript.parent).st_mode & 0o777 == 0o700
    assert os.stat(state_dir / "transcripts").st_mode & 0o777 == 0o700


# -- AC-4: no arguments, summaries or output ----------------------------------


async def test_tool_arguments_summaries_and_output_are_not_saved(meeting_ws):
    turns = [
        [ToolCall("write", {"path": "notes.md", "content": "BODY-4c1e of the note"}), Text("Written.")],
        [ToolCall("shell", {"command": "echo CMD-9f2a"}), Text("Ran it.")],
        [ToolCall("fetch", {"url": "https://host-7b3d.example/page"}), Text("Could not fetch.")],
    ]
    store = FileSessionStore(meeting_ws)
    agent = make_agent(
        meeting_ws,
        turns,
        store,
        tools=["write", "shell", "fetch"],
        permission_mode="bypass",
        permissions={"fetch": "deny"},  # recorded, but never leaves the machine
    )
    session = agent.new_session()
    for prompt in ("save a note", "run the command", "get the page"):
        await session.run(prompt)
    await session.close()

    # The in-memory records do carry them, so the check below means something.
    kept = json.dumps([[vars(c) for c in t.tool_calls] for t in session.history], default=str)
    for secret in ("BODY-4c1e", "CMD-9f2a", "host-7b3d"):
        assert secret in kept
    text = store.path_for(session.id).read_text(encoding="utf-8")
    for secret in ("BODY-4c1e", "CMD-9f2a", "host-7b3d"):
        assert secret not in text
    records = lines_of(store.path_for(session.id))
    assert [r["tools"] for r in records] == [
        [{"name": "write", "status": "ok"}],
        [{"name": "shell", "status": "ok"}],
        [{"name": "fetch", "status": "denied"}],
    ]
    assert all(set(r) == {"v", "t", "prompt", "response", "stop_reason", "tools"} for r in records)


# -- AC-5: opt-in, and a store that fails does not fail the turn ---------------


async def test_nothing_is_saved_without_a_store(meeting_ws, state_dir):
    session = make_agent(meeting_ws, ["ok"]).new_session()
    await session.run("hi")
    await session.close()
    assert not (state_dir / "transcripts").exists()
    assert session.status()["resumed"] is False


async def test_a_store_that_cannot_write_does_not_break_the_turn(meeting_ws, state_dir, caplog):
    state_dir.mkdir(parents=True)
    (state_dir / "transcripts").write_text("not a directory")
    session = make_agent(meeting_ws, ["still answered"], FileSessionStore(meeting_ws)).new_session()
    with caplog.at_level(logging.WARNING):
        result = await session.run("hi")
    await session.close()
    assert result.text == "still answered" and result.stop_reason == "end_turn"
    assert len(session.history) == 1
    assert any("was not saved" in r.getMessage() for r in caplog.records)


async def test_a_transcript_that_cannot_be_written_does_not_break_the_turn(meeting_ws, caplog):
    store = FileSessionStore(meeting_ws)
    session = make_agent(meeting_ws, ["still answered"], store).new_session()
    store.path_for(session.id).mkdir()  # something in the way of the file
    with caplog.at_level(logging.WARNING):
        result = await session.run("hi")
    await session.close()
    assert result.text == "still answered"
    assert any("was not saved" in r.getMessage() for r in caplog.records)


async def test_a_read_only_store_resumes_without_appending(meeting_ws):
    session_id = await save_two_turns(meeting_ws, FileSessionStore(meeting_ws))
    reader = FileSessionStore(meeting_ws, persist=False)
    session = make_agent(meeting_ws, ["ok"], reader).new_session(session_id=session_id)
    await session.run("a third question")
    await session.close()
    assert len(lines_of(reader.path_for(session_id))) == 2


@pytest.mark.parametrize("value", [True, False])
def test_the_sessions_persist_key(value):
    assert KennelConfig().persist_sessions is False
    assert apply_config(KennelConfig(), {"sessions": {"persist": value}}, "<test>").persist_sessions is value


def test_a_bad_sessions_persist_key_is_rejected():
    with pytest.raises(ConfigurationError, match="sessions.persist must be true or false"):
        apply_config(KennelConfig(), {"sessions": {"persist": "yes"}}, "<test>")
    with pytest.raises(ConfigurationError, match="'sessions' must be an object"):
        apply_config(KennelConfig(), {"sessions": True}, "<test>")


# -- AC-6: session grants stay with the process --------------------------------


async def test_a_session_grant_is_not_carried_into_the_resumed_session(meeting_ws):
    asked: list[str] = []

    def prompter(request):
        asked.append(request.tool_name)
        return Approval.SESSION

    write = [ToolCall("write", {"path": "a.md", "content": "x"}), Text("done")]
    store = FileSessionStore(meeting_ws)
    first = make_agent(meeting_ws, [write, write], store, tools=["write"], prompter=prompter).new_session()
    await first.run("write a")
    await first.run("write a again")
    await first.close()
    assert asked == ["write"]  # the grant covered the second call in the same process

    # A later process: a new Agent, so a new PermissionManager.
    again = make_agent(meeting_ws, [write], store, tools=["write"], prompter=prompter)
    resumed = again.new_session(session_id=first.id)
    assert resumed.resumed
    await resumed.run("write a once more")
    await resumed.close()
    assert asked == ["write", "write"]  # asked again


# -- AC-7: /clear deletes the saved conversation ---------------------------------


async def test_clear_deletes_the_transcript(meeting_ws):
    store = FileSessionStore(meeting_ws)
    session_id = await save_two_turns(meeting_ws, store)
    session = make_agent(meeting_ws, ["fresh"], store).new_session(session_id=session_id)
    await session.clear()
    assert not store.path_for(session_id).exists()
    assert session.id == session_id and session.history == [] and not session.resumed
    await session.run("start over")  # the next turn starts a new file under the same id
    await session.close()
    assert [line["prompt"] for line in lines_of(store.path_for(session_id))] == ["start over"]


async def test_a_cleared_conversation_does_not_come_back(meeting_ws):
    store = FileSessionStore(meeting_ws)
    session_id = await save_two_turns(meeting_ws, store)
    session = make_agent(meeting_ws, [], store).new_session(session_id=session_id)
    await session.clear()
    await session.close()
    again = make_agent(meeting_ws, [], store).new_session(session_id=session_id)
    assert again.history == [] and not again.resumed
    await again.close()


async def test_clear_raises_when_the_transcript_cannot_be_deleted(meeting_ws):
    store = FileSessionStore(meeting_ws)
    session_id = await save_two_turns(meeting_ws, store)
    session = make_agent(meeting_ws, [], store).new_session(session_id=session_id)
    os.chmod(store.directory, 0o500)
    try:
        with pytest.raises(SessionError, match="cannot delete"):
            await session.clear()
        assert len(session.history) == 2  # nothing was forgotten
    finally:
        os.chmod(store.directory, 0o700)
        await session.close()
    assert store.path_for(session_id).exists()


# -- AC-8: ids, locking and damaged files ----------------------------------------


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "a" * 65, "a.b", "ü"])
async def test_an_invalid_id_is_refused(meeting_ws, bad):
    agent = make_agent(meeting_ws, [], FileSessionStore(meeting_ws))
    with pytest.raises(ConfigurationError, match="invalid session id"):
        agent.new_session(session_id=bad)


async def test_a_64_character_id_is_fine(meeting_ws):
    session = make_agent(meeting_ws, [], FileSessionStore(meeting_ws)).new_session(session_id="a" * 64)
    await session.close()


async def test_an_open_id_cannot_be_opened_twice(meeting_ws):
    store = FileSessionStore(meeting_ws)
    agent = make_agent(meeting_ws, [], store)
    first = agent.new_session(session_id="shared")
    with pytest.raises(ConfigurationError, match="in use"):
        agent.new_session(session_id="shared")
    await first.close()
    second = agent.new_session(session_id="shared")  # released by close()
    await second.close()


def test_a_cut_short_last_line_is_dropped_and_repaired(meeting_ws, caplog):
    store = FileSessionStore(meeting_ws)
    store.directory.mkdir(parents=True)
    good = {"v": 1, "t": 1.0, "prompt": "p", "response": "r", "stop_reason": "end_turn", "tools": []}
    store.path_for("s1").write_text(json.dumps(good) + "\n" + '{"v": 1, "prompt": "half', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert [t.prompt for t in store.load("s1")] == ["p"]
    assert any("cut short" in r.getMessage() for r in caplog.records)

    from kennel.session import Turn

    store.append("s1", Turn("p2", "r2", "end_turn"))  # the next write does not glue onto it
    assert [t.prompt for t in store.load("s1")] == ["p", "p2"]


def test_a_broken_line_in_the_middle_is_an_error(meeting_ws):
    store = FileSessionStore(meeting_ws)
    store.directory.mkdir(parents=True)
    good = json.dumps({"v": 1, "t": 1.0, "prompt": "p", "response": "r", "stop_reason": "end_turn", "tools": []})
    store.path_for("s1").write_text(good + "\n{broken\n" + good + "\n", encoding="utf-8")
    with pytest.raises(SessionError, match=":2: the saved conversation is corrupt"):
        store.load("s1")


def test_an_unknown_version_is_an_error(meeting_ws):
    store = FileSessionStore(meeting_ws)
    store.directory.mkdir(parents=True)
    line = {"v": 2, "t": 1.0, "prompt": "p", "response": "r", "stop_reason": "end_turn", "tools": []}
    store.path_for("s1").write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(SessionError, match="unsupported transcript version 2"):
        store.load("s1")


async def test_a_failed_load_releases_the_lock(meeting_ws):
    store = FileSessionStore(meeting_ws)
    store.directory.mkdir(parents=True)
    store.path_for("s1").write_text('{"v": 9}\n', encoding="utf-8")
    agent = make_agent(meeting_ws, [], store)
    with pytest.raises(SessionError):
        agent.new_session(session_id="s1")
    store.path_for("s1").unlink()
    await agent.new_session(session_id="s1").close()  # not "in use"


# -- AC-9: a workspace only sees its own conversations ---------------------------


async def test_latest_and_load_stay_inside_the_workspace(tmp_path):
    ws_a, ws_b = tmp_path / "a", tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()
    store_a, store_b = FileSessionStore(ws_a), FileSessionStore(ws_b)
    assert store_a.latest() is None

    older = make_agent(ws_a, ["1"], store_a).new_session(session_id="older")
    await older.run("first")
    await older.close()
    newer = make_agent(ws_a, ["2"], store_a).new_session(session_id="newer")
    await newer.run("second")
    await newer.close()
    os.utime(store_a.path_for("older"), (1_000, 1_000))
    other = make_agent(ws_b, ["3"], store_b).new_session(session_id="elsewhere")
    await other.run("third")
    await other.close()  # the most recent file overall, but in another workspace

    assert store_a.latest() == "newer"
    assert store_b.latest() == "elsewhere"
    assert store_a.load("elsewhere") == []
    resumed = make_agent(ws_a, [], store_a).new_session(session_id="elsewhere")
    assert not resumed.resumed
    await resumed.close()
