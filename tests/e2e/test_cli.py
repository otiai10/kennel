"""End-to-end CLI runs with the scripted mock provider (no Apple model needed)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TRANSCRIPT_FLOW = {
    "turns": [
        [
            {"tool": "glob", "arguments": {"pattern": "transcripts/*.txt"}},
            {"tool": "read", "arguments": {"path": "transcripts/2026-09-16.txt", "start_line": 1, "end_line": 12}},
            {"tool": "read", "arguments": {"path": "../outside.txt"}},
            {"tool": "write", "arguments": {"path": "minutes/2026-09-16.md", "content": "# Minutes\n"}},
            {"text": "Decision: ship v0.1 on Friday. TODO: README security section (Bob). Unresolved: release notes owner."},
        ]
    ]
}


def run_cli(args, ws: Path, script=None, stdin="", env=None):
    e = {**os.environ, "NO_COLOR": "1", "KENNEL_MOCK_SCRIPT": json.dumps(script or TRANSCRIPT_FLOW)}
    e.update(env or {})
    proc = subprocess.run(
        [sys.executable, "-m", "kennel.cli.main", str(ws), "--provider", "mock", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=e,
        timeout=60,
    )
    return proc


def test_help_and_version(meeting_ws):
    assert run_cli(["--help"], meeting_ws).returncode == 0
    p = run_cli(["--version"], meeting_ws)
    assert p.returncode == 0 and "kennel 0.0.3" in p.stdout + p.stderr


def test_one_shot_transcript_flow(meeting_ws):
    p = run_cli(["-p", "summarize the latest meeting", "--verbose"], meeting_ws)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "● Glob transcripts/*.txt" in out
    assert "● Read transcripts/2026-09-16.txt [1-12]" in out
    assert "✗ Path escapes the workspace" in out
    assert "⊘ denied: Write minutes/2026-09-16.md" in out  # stdin is not a tty: ask => deny
    assert out.rstrip().endswith("Unresolved: release notes owner.")
    assert not (meeting_ws / "minutes").exists()
    assert "↳" in out  # verbose sizes


def test_read_only_removes_mutation_tools(meeting_ws):
    p = run_cli(["-p", "x", "--read-only"], meeting_ws)
    assert p.returncode == 0
    assert "unknown tool 'write'" in p.stdout
    assert not (meeting_ws / "minutes").exists()


def test_allow_write_flag_writes(meeting_ws):
    p = run_cli(["-p", "x", "--allow-write"], meeting_ws)
    assert p.returncode == 0, p.stderr
    assert "● Write minutes/2026-09-16.md" in p.stdout
    assert (meeting_ws / "minutes" / "2026-09-16.md").read_text() == "# Minutes\n"


def test_instructions_missing_file_exits_2(meeting_ws):
    p = run_cli(["-p", "x", "--instructions", "@nope.md"], meeting_ws)
    assert p.returncode == 2 and "--instructions" in p.stderr and "nope.md" in p.stderr


def test_conflicting_flags(meeting_ws):
    p = run_cli(["-p", "x", "--read-only", "--allow-write"], meeting_ws)
    assert p.returncode == 2 and "cannot be combined" in p.stderr


def test_trace_emits_json_events(meeting_ws):
    p = run_cli(["-p", "x", "--trace"], meeting_ws)
    events = [json.loads(line) for line in p.stderr.splitlines() if line.startswith("{")]
    types = [e["type"] for e in events]
    assert types[:3] == ["session.started", "model.started", "tool.requested"]
    assert "permission.denied" in types and types[-1] == "session.completed"
    assert not any("content" in e for e in events)


def test_interactive_session(meeting_ws):
    script = {"turns": [[{"tool": "glob", "arguments": {"pattern": "*.md"}}, {"text": "answer one"}], "answer two"]}
    stdin = "/help\n/status\nfirst\n/clear\nsecond\n/unknown\n/exit\n"
    p = run_cli([], meeting_ws, script=script, stdin=stdin)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert out.startswith("Kennel v0.0.3\nworkspace: ")
    assert "model: MockModel\nmode: local\ntools: glob, grep, read, write (ask), edit (ask), shell (ask)" in out
    assert "/clear    forget the conversation" in out
    assert "turns: 0" in out and "permissions: edit=ask" in out
    assert "● Glob *.md" in out and "answer one" in out
    assert "(conversation cleared)" in out and "answer two" in out
    assert "unknown command /unknown" in out


def test_interactive_eof_exits_cleanly(meeting_ws):
    p = run_cli([], meeting_ws, script={"turns": []}, stdin="")
    assert p.returncode == 0 and p.stdout.startswith("Kennel")


def test_workspace_missing(tmp_path):
    p = run_cli(["-p", "x"], tmp_path / "nope")
    assert p.returncode == 1 and "Workspace does not exist" in p.stderr


def test_project_config_is_applied(meeting_ws):
    (meeting_ws / "kennel.json").write_text(json.dumps({"permissions": {"write": "allow"}, "agent": {"max_tool_calls": 2}}))
    p = run_cli(["-p", "x"], meeting_ws)
    assert p.returncode == 0, p.stderr
    assert "tool call limit (2 per turn) reached" in p.stdout
    assert "(tool call limit reached" in p.stdout


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty needed")
def test_permission_prompt_on_tty(meeting_ws):
    """With a pty, the write asks for approval; answering 'y' lets it through."""
    import pty
    import select
    import time

    script = {"turns": [[{"tool": "write", "arguments": {"path": "ok.md", "content": "yes\n"}}, {"text": "wrote it"}]]}
    env = {**os.environ, "NO_COLOR": "1", "KENNEL_MOCK_SCRIPT": json.dumps(script)}
    pid, fd = pty.fork()
    if pid == 0:  # child
        os.chdir(meeting_ws)
        os.execvpe(sys.executable, [sys.executable, "-m", "kennel.cli.main", ".", "--provider", "mock", "-p", "write"], env)
    output = b""
    deadline = time.time() + 30
    answered = False
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.2)
        if r:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
            if b"[y] once" in output and not answered:
                os.write(fd, b"y\n")
                answered = True
        _, status = os.waitpid(pid, os.WNOHANG)
        if status:
            break
    text = output.decode(errors="replace")
    assert "Allow Write: Write ok.md" in text and "Creates new file ok.md" in text
    assert "● Write ok.md" in text and "wrote it" in text
    assert (meeting_ws / "ok.md").read_text() == "yes\n"


def test_closed_stdout_exits_quietly(meeting_ws):
    """`kennel ... | head` must not print a traceback when the pipe closes."""
    e = {**os.environ, "NO_COLOR": "1", "KENNEL_MOCK_SCRIPT": json.dumps({"turns": ["x" * 20000]})}
    proc = subprocess.Popen(
        [sys.executable, "-m", "kennel.cli.main", str(meeting_ws), "--provider", "mock", "-p", "go"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e,
    )
    proc.stdout.read(10)
    proc.stdout.close()
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    assert b"Traceback" not in err and b"Broken pipe" not in err
