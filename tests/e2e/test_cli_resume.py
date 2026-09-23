"""``--persist`` / ``--no-persist`` / ``--continue`` / ``--resume`` end to end (#62), on the mock provider.

``HOME`` points at a tmp directory so a developer's own ``~/.config/kennel/settings.json`` cannot
turn saving on behind a test's back; ``KENNEL_STATE_DIR`` is the autouse tmp state directory.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from kennel import FileSessionStore


def kennel(ws: Path, home: Path, *args: str, answers=("ok",), stdin: str = "") -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "HOME": str(home),
        "KENNEL_MOCK_SCRIPT": json.dumps({"turns": list(answers)}),
    }
    return subprocess.run(
        [sys.executable, "-m", "kennel.cli.main", str(ws), "--provider", "mock", "--no-log", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


def transcripts(state_dir: Path) -> list[Path]:
    return sorted((state_dir / "transcripts").glob("*/*.jsonl")) if state_dir.exists() else []


def prompts(path: Path) -> list[str]:
    return [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines()]


def session_id_of(proc: subprocess.CompletedProcess) -> str:
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["session_id"]


# -- AC-5: opt-in, flags over the config key -------------------------------------


def test_nothing_is_saved_by_default(meeting_ws, home, state_dir):
    assert kennel(meeting_ws, home, "-p", "hello").returncode == 0
    assert transcripts(state_dir) == []


def test_persist_saves_one_shot_runs(meeting_ws, home, state_dir):
    proc = kennel(meeting_ws, home, "-p", "hello", "--persist", "--output-format", "json")
    session_id = session_id_of(proc)
    (path,) = transcripts(state_dir)
    assert path.stem == session_id and prompts(path) == ["hello"]


def test_the_config_key_saves_and_no_persist_wins(meeting_ws, home, state_dir):
    (meeting_ws / "kennel.json").write_text(json.dumps({"sessions": {"persist": True}}))
    assert kennel(meeting_ws, home, "-p", "hello", "--no-persist").returncode == 0
    assert transcripts(state_dir) == []
    assert kennel(meeting_ws, home, "-p", "hello").returncode == 0
    assert len(transcripts(state_dir)) == 1


def test_the_user_config_key_saves_too(meeting_ws, home, state_dir):
    settings = home / ".config" / "kennel" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"sessions": {"persist": True}}))
    assert kennel(meeting_ws, home, "-p", "hello").returncode == 0
    assert len(transcripts(state_dir)) == 1


def test_persist_and_no_persist_cannot_be_combined(meeting_ws, home):
    proc = kennel(meeting_ws, home, "-p", "x", "--persist", "--no-persist")
    assert proc.returncode == 2 and "not allowed with" in proc.stderr


# -- AC-9 and the resume flags ---------------------------------------------------


def test_continue_resumes_the_latest_conversation_of_this_workspace(meeting_ws, home, state_dir, tmp_path):
    first = session_id_of(kennel(meeting_ws, home, "-p", "first", "--persist", "--output-format", "json"))
    other_ws = tmp_path / "other"
    other_ws.mkdir()
    # Saved later, but in another workspace: --continue must not pick it.
    session_id_of(kennel(other_ws, home, "-p", "elsewhere", "--persist", "--output-format", "json"))

    proc = kennel(meeting_ws, home, "--continue", "-p", "second", "--persist", "--output-format", "json")
    assert session_id_of(proc) == first
    assert prompts(FileSessionStore(meeting_ws).path_for(first)) == ["first", "second"]


def test_resuming_while_saving_is_off_reads_but_does_not_append(meeting_ws, home):
    first = session_id_of(kennel(meeting_ws, home, "-p", "first", "--persist", "--output-format", "json"))
    proc = kennel(meeting_ws, home, "--resume", first, "-p", "second", "--output-format", "json")
    assert session_id_of(proc) == first
    assert prompts(FileSessionStore(meeting_ws).path_for(first)) == ["first"]


def test_resume_shows_resumed_in_status(meeting_ws, home):
    first = session_id_of(kennel(meeting_ws, home, "-p", "first", "--persist", "--output-format", "json"))
    proc = kennel(meeting_ws, home, "--resume", first, stdin="/status\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert f"session_id: {first}\nturns: 1\n" in proc.stdout
    assert "resumed: True" in proc.stdout
    fresh = kennel(meeting_ws, home, stdin="/status\n/exit\n")
    assert "resumed: False" in fresh.stdout


def test_resume_does_not_find_another_workspaces_conversation(meeting_ws, home, tmp_path):
    other_ws = tmp_path / "other"
    other_ws.mkdir()
    theirs = session_id_of(kennel(other_ws, home, "-p", "elsewhere", "--persist", "--output-format", "json"))
    proc = kennel(meeting_ws, home, "--resume", theirs, "-p", "x")
    assert proc.returncode == 2 and f"no saved conversation '{theirs}' in this workspace" in proc.stderr


def test_continue_with_nothing_saved_is_an_error(meeting_ws, home):
    proc = kennel(meeting_ws, home, "--continue", "-p", "x")
    assert proc.returncode == 2 and "--continue: no saved conversation in this workspace" in proc.stderr


def test_continue_and_resume_cannot_be_combined(meeting_ws, home):
    proc = kennel(meeting_ws, home, "--continue", "--resume", "abc", "-p", "x")
    assert proc.returncode == 2 and "not allowed with" in proc.stderr


# -- AC-8 -------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../x", "a" * 65])
def test_a_bad_resume_id_is_a_configuration_error(meeting_ws, home, bad):
    proc = kennel(meeting_ws, home, "--resume", bad, "-p", "x")
    assert proc.returncode == 2 and "invalid session id" in proc.stderr


def test_a_session_open_in_another_process_is_refused(meeting_ws, home):
    first = session_id_of(kennel(meeting_ws, home, "-p", "first", "--persist", "--output-format", "json"))
    release = FileSessionStore(meeting_ws).lock(first)  # this test process holds it
    try:
        proc = kennel(meeting_ws, home, "--resume", first, "-p", "x")
    finally:
        release()
    assert proc.returncode == 2 and f"session {first} is in use (open in another session or process)" in proc.stderr
    assert kennel(meeting_ws, home, "--resume", first, "-p", "x").returncode == 0  # free again


# -- AC-7 ---------------------------------------------------------------------------


def test_clear_deletes_the_saved_conversation(meeting_ws, home):
    first = session_id_of(kennel(meeting_ws, home, "-p", "first", "--persist", "--output-format", "json"))
    path = FileSessionStore(meeting_ws).path_for(first)
    proc = kennel(meeting_ws, home, "--resume", first, stdin="/clear\n/exit\n")
    assert proc.returncode == 0 and "(conversation cleared)" in proc.stdout
    assert not path.exists()
    again = kennel(meeting_ws, home, "--resume", first, "-p", "x")
    assert again.returncode == 2 and "no saved conversation" in again.stderr


@pytest.mark.skipif(not hasattr(stat, "UF_IMMUTABLE") or not hasattr(os, "chflags"), reason="needs chflags")
def test_clear_that_cannot_delete_says_so(meeting_ws, home):
    first = session_id_of(kennel(meeting_ws, home, "-p", "first", "--persist", "--output-format", "json"))
    store = FileSessionStore(meeting_ws)
    # The store keeps its directory 0700 itself, so an immutable file is what makes unlink fail.
    os.chflags(store.path_for(first), stat.UF_IMMUTABLE)
    try:
        proc = kennel(meeting_ws, home, "--resume", first, stdin="/clear\n/exit\n")
    finally:
        os.chflags(store.path_for(first), 0)
    assert "(conversation cleared)" not in proc.stdout
    assert "cannot delete the saved conversation" in proc.stderr
    assert store.path_for(first).exists()
