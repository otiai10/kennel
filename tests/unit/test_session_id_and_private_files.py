"""One rule for session ids and one way to create a session's files (issue #83).

The id names the event log and the transcript, so it is checked when the session is created,
with or without a store; both files are ``0600`` in ``0700`` directories whatever the umask.
"""

import os
from pathlib import Path

import pytest

from kennel import (
    Agent,
    ConfigurationError,
    FileSessionStore,
    JsonlEventLog,
    KennelConfig,
    MockProvider,
    _statefiles,
    events,
    session_log_path,
    sessions,
)
from kennel.providers.mock import Text


def all_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


def logging_agent(ws, store=None) -> Agent:
    return Agent(
        ws,
        provider=MockProvider([[Text("hi")]]),
        config=KennelConfig(log_events=True),
        session_store=store,
    )


@pytest.mark.parametrize("bad", ["../../x", "a/b", "..", "", "a" * 65, "with space", "a.b"])
async def test_a_bad_id_is_refused_without_a_store_and_nothing_is_written(meeting_ws, state_dir, tmp_path, bad):
    """AC-1: the SDK path without a store checks the id before any path is built."""
    before = all_files(tmp_path)
    with pytest.raises(ConfigurationError, match="invalid session id"):
        logging_agent(meeting_ws).new_session(session_id=bad)
    assert all_files(tmp_path) == before
    assert not state_dir.exists()


@pytest.mark.parametrize("good", ["abc123", "A-b_c", "x" * 64])
async def test_a_good_id_is_accepted_with_and_without_a_store(meeting_ws, state_dir, good):
    """AC-2: ids that were valid stay valid, whether or not conversations are saved."""
    plain = logging_agent(meeting_ws).new_session(session_id=good)
    assert plain.id == good
    await plain.run("hello")
    await plain.close()
    assert (state_dir / "sessions" / f"{good}.jsonl").exists()

    stored = logging_agent(meeting_ws, FileSessionStore(meeting_ws)).new_session(session_id=good)
    assert stored.id == good
    await stored.close()


def test_an_id_is_still_generated_when_none_is_given(meeting_ws):
    """AC-2: no id means a fresh random one, exactly as before."""
    first, second = (logging_agent(meeting_ws).new_session() for _ in range(2))
    assert first.id != second.id
    assert len(first.id) == 12 and _statefiles.SESSION_ID_PATTERN.fullmatch(first.id)


def test_session_log_path_refuses_a_bad_id():
    with pytest.raises(ConfigurationError):
        session_log_path("../../x")


@pytest.fixture
def loose_umask():
    previous = os.umask(0)
    try:
        yield
    finally:
        os.umask(previous)


def mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


async def test_the_event_log_is_owner_only_under_a_loose_umask(meeting_ws, state_dir, loose_umask):
    """AC-3: the log file is 0600 and every directory it needed is 0700, umask or not."""
    session = logging_agent(meeting_ws).new_session()
    await session.run("hello")
    await session.close()
    log = state_dir / "sessions" / f"{session.id}.jsonl"
    assert mode(log) == 0o600
    assert mode(log.parent) == 0o700
    assert mode(state_dir) == 0o700


def test_an_existing_event_log_file_is_tightened(tmp_path, loose_umask):
    """AC-3: a file that was already there, readable by others, becomes 0600."""
    path = tmp_path / "log.jsonl"
    path.write_text("")
    os.chmod(path, 0o644)
    JsonlEventLog(path).close()
    assert mode(path) == 0o600


def test_existing_directories_the_log_did_not_create_are_left_alone(tmp_path, loose_umask):
    """An SDK caller may point the log anywhere; Kennel only owns what it creates."""
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o755)
    JsonlEventLog(shared / "new" / "log.jsonl").close()
    assert mode(shared) == 0o755
    assert mode(shared / "new") == 0o700


async def test_one_validator_and_one_opener_serve_both_files(meeting_ws, monkeypatch):
    """AC-4: sessions.py and events.py share the id rule and the private-file opener."""
    assert sessions.check_session_id is _statefiles.check_session_id
    assert events.check_session_id is _statefiles.check_session_id
    opened: list[str] = []
    real = _statefiles.open_private

    def counting(path, flags, **kw):
        opened.append(Path(path).suffix)
        return real(path, flags, **kw)

    monkeypatch.setattr(sessions, "open_private", counting)
    monkeypatch.setattr(events, "open_private", counting)
    session = logging_agent(meeting_ws, FileSessionStore(meeting_ws)).new_session()
    await session.run("hello")
    await session.close()
    assert {".lock", ".jsonl"} <= set(opened)
    assert opened.count(".jsonl") >= 2  # the transcript and the event log
