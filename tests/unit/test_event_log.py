"""The session event log: where it goes, what it holds and how it fails.

The log is a local file and nothing else (principle 1), it carries no file or generated
content (principle 3), and a log that cannot be written must not stop the agent.
"""

import json
import os
from pathlib import Path

import pytest

from kennel import (
    Agent,
    Event,
    EventBus,
    JsonlEventLog,
    KennelConfig,
    MockProvider,
    event_json,
    session_log_path,
    state_dir,
)
from kennel.providers.mock import Text, ToolCall

TURNS = [[ToolCall("read", {"path": "transcripts/2026-09-16.txt"}), Text("summarized")]]


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# -- path resolution ----------------------------------------------------------


def test_kennel_state_dir_wins_over_xdg_and_home(monkeypatch, tmp_path):
    monkeypatch.setenv("KENNEL_STATE_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert state_dir() == tmp_path / "explicit"
    assert session_log_path("abc123") == tmp_path / "explicit" / "sessions" / "abc123.jsonl"


def test_xdg_state_home_then_the_default_location(monkeypatch, tmp_path):
    monkeypatch.delenv("KENNEL_STATE_DIR")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert state_dir() == tmp_path / "xdg" / "kennel"
    monkeypatch.delenv("XDG_STATE_HOME")
    assert state_dir() == Path.home() / ".local" / "state" / "kennel"


# -- the subscriber itself ----------------------------------------------------


def test_event_json_is_the_flat_form_shared_with_trace():
    record = json.loads(event_json(Event("tool.completed", "sid", {"tool": "read", "output_bytes": 12})))
    assert record == {"t": record["t"], "type": "tool.completed", "session_id": "sid", "tool": "read", "output_bytes": 12}


def test_it_only_writes_its_own_session_and_closes_as_a_context_manager(tmp_path):
    path = tmp_path / "log.jsonl"
    bus = EventBus()
    with JsonlEventLog(path, session_id="mine") as log:
        bus.subscribe(log)
        bus.emit("model.started", "mine", prompt_chars=3)
        bus.emit("model.started", "theirs", prompt_chars=3)
        assert log.enabled
    assert [line["session_id"] for line in read_lines(path)] == ["mine"]
    assert not log.enabled


def test_an_unwritable_directory_disables_the_log_with_a_warning(tmp_path, caplog):
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o500)
    try:
        log = JsonlEventLog(blocked / "sub" / "log.jsonl")
        assert not log.enabled
        log(Event("model.started", "sid", {}))  # silently does nothing
        assert any("event log disabled" in r.getMessage() for r in caplog.records)
    finally:
        os.chmod(blocked, 0o700)


def test_the_directory_is_created_for_the_owner_only(tmp_path):
    path = tmp_path / "state" / "sessions" / "s.jsonl"
    JsonlEventLog(path).close()
    assert path.exists() and (os.stat(path.parent).st_mode & 0o777) == 0o700


# -- wired into a session -----------------------------------------------------


def agent_for(meeting_ws, **config_kw) -> Agent:
    return Agent(meeting_ws, provider=MockProvider(list(TURNS)), config=KennelConfig(**config_kw))


async def test_a_logging_session_records_started_through_completed(meeting_ws, state_dir):
    """AC-1: every line of the turn, in JSON Lines, each with t / type / session_id."""
    agent = agent_for(meeting_ws, log_events=True)
    session = agent.new_session()
    assert session.log_path == state_dir / "sessions" / f"{session.id}.jsonl"
    assert not session.log_path.exists()  # nothing is written until the session runs
    await session.run("summarize it")
    await session.close()
    lines = read_lines(session.log_path)
    types = [line["type"] for line in lines]
    assert types[0] == "session.started" and types[-1] == "session.completed"
    assert "tool.completed" in types and "model.completed" in types
    assert all({"t", "type", "session_id"} <= set(line) for line in lines)
    assert {line["session_id"] for line in lines} == {session.id}
    assert all(isinstance(line["t"], float) for line in lines)


async def test_the_log_carries_no_file_or_generated_content(meeting_ws):
    """AC-2: sizes and summaries only; model.delta is a character count."""
    agent = agent_for(meeting_ws, log_events=True)
    session = agent.new_session()
    await session.run("summarize it", on_delta=lambda _: None)
    await session.close()
    lines = read_lines(session.log_path)
    assert not any("text" in line or "content" in line for line in lines)
    deltas = [line for line in lines if line["type"] == "model.delta"]
    assert deltas and all(set(line) == {"t", "type", "session_id", "chars"} for line in deltas)
    body = session.log_path.read_text(encoding="utf-8")
    assert "summarized" not in body  # the answer
    assert "Ship v0.1" not in body  # the transcript the tool read


async def test_a_broken_log_does_not_break_the_turn(meeting_ws, state_dir, caplog):
    """AC-3: the run succeeds and the failure is reported through logging."""
    state_dir.mkdir(parents=True)
    (state_dir / "sessions").write_text("not a directory")  # make the log path unusable
    agent = agent_for(meeting_ws, log_events=True)
    session = agent.new_session()
    result = await session.run("summarize it")
    await session.close()
    assert result.text == "summarized" and result.stop_reason == "end_turn"
    assert any("event log disabled" in r.getMessage() for r in caplog.records)


async def test_logging_is_opt_in_for_the_sdk(meeting_ws, state_dir):
    """An embedding application is not made to write files it did not ask for."""
    session = agent_for(meeting_ws).new_session()  # log_events unset
    assert session.log_path is None and session.status()["log"] == "off"
    await session.run("summarize it")
    await session.close()
    assert not state_dir.exists()


async def test_status_reports_the_log_path(meeting_ws, state_dir):
    """AC-5, the part /status is responsible for."""
    session = agent_for(meeting_ws, log_events=True).new_session()
    assert session.status()["log"] == str(state_dir / "sessions" / f"{session.id}.jsonl")


@pytest.mark.parametrize("value", [True, False])
def test_the_config_key_round_trips(tmp_path, value):
    from kennel.config import apply_config

    cfg = apply_config(KennelConfig(), {"logging": {"events": value}}, "<test>")
    assert cfg.log_events is value


def test_a_bad_config_key_is_rejected():
    from kennel.config import apply_config
    from kennel.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="logging.events must be true or false"):
        apply_config(KennelConfig(), {"logging": {"events": "yes"}}, "<test>")
    with pytest.raises(ConfigurationError, match="'logging' must be an object"):
        apply_config(KennelConfig(), {"logging": True}, "<test>")
