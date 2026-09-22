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
    assert p.returncode == 0 and "kennel 0.1.0" in p.stdout + p.stderr


def test_readme_explains_the_estimated_fallback_for_reporting_providers():
    """AC-3 (#57): the doc-only fix must be pinned in-repo, not just eyeballed once.

    Both the llama-server and llama-cpp sections promise the same thing: right after
    interrupt/timeout/failure/compact retires the live provider session, `/usage`
    falls back to *estimated* even though the provider otherwise reports real counts.
    """
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    assert "`/usage` briefly switches" in readme  # llama-server section
    assert "As with `llama-server`, that flips back to" in readme  # llama-cpp section


def test_one_shot_transcript_flow(meeting_ws):
    p = run_cli(["-p", "summarize the latest meeting", "--verbose"], meeting_ws)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "● Glob transcripts/*.txt" in out
    assert "● Read transcripts/2026-09-16.txt [1-12]" in out
    assert "✗ Path escapes the workspace" in out
    assert "⊘ denied: Write minutes/2026-09-16.md" in out  # stdin is not a tty: ask => deny
    assert "Unresolved: release notes owner." in out
    assert out.rstrip().splitlines()[-1].startswith("(context: ")  # window usage, now shown anyway
    assert not (meeting_ws / "minutes").exists()
    assert "in 0 ms" in out or " ms" in out  # --verbose adds the timing to the size line


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


BIG_READ = {
    "turns": [
        [
            {"tool": "read", "arguments": {"path": "big.txt"}},
            {"text": "done"},
        ]
    ]
}


def test_tool_sizes_and_the_context_line_are_shown_by_default(meeting_ws):
    """AC-3 (and AC-2 through the real CLI): no --verbose needed."""
    (meeting_ws / "big.txt").write_text("x" * 20_000)
    p = run_cli(["-p", "read it"], meeting_ws, script=BIG_READ)
    assert p.returncode == 0, p.stderr
    lines = p.stdout.rstrip().splitlines()
    assert "● Read big.txt" in lines[0]
    assert "bytes" in lines[1] and "exceeds the 4,096-token window" in lines[1]
    assert lines[-1].startswith("(context: ") and "estimated" in lines[-1]
    assert " ms" not in lines[1]  # the timing is still --verbose only


def test_the_context_line_appears_after_every_interactive_turn(meeting_ws):
    """AC-3: one line per turn in the REPL, without --verbose."""
    script = {"turns": ["one", "two"]}
    p = run_cli([], meeting_ws, script=script, stdin="first\nsecond\n/exit\n")
    assert p.returncode == 0, p.stderr
    contexts = [line for line in p.stdout.splitlines() if line.startswith("(context: ")]
    assert len(contexts) == 2, p.stdout
    used = [int(line.split("(")[-1].split(" tokens")[0]) for line in contexts]
    assert used[1] > used[0]  # the window fills up as the conversation grows


def test_machine_output_stays_json_only(meeting_ws):
    """AC-3: quiet suppresses both new lines, so stdout is still JSON alone."""
    (meeting_ws / "big.txt").write_text("x" * 20_000)
    p = run_cli(["-p", "read it", "--output-format", "json"], meeting_ws, script=BIG_READ)
    assert p.returncode == 0, p.stderr
    assert "(context" not in p.stdout and "↳" not in p.stdout
    json.loads(p.stdout)  # a single JSON document, nothing else


def test_the_cli_keeps_a_session_event_log_by_default(meeting_ws, state_dir):
    """AC-4: KENNEL_STATE_DIR/sessions/<session_id>.jsonl after a one-shot run."""
    p = run_cli(["-p", "x", "--output-format", "json"], meeting_ws)
    assert p.returncode == 0, p.stderr
    session_id = json.loads(p.stdout)["session_id"]
    log = state_dir / "sessions" / f"{session_id}.jsonl"
    assert log.is_file(), sorted((state_dir / "sessions").glob("*")) if state_dir.exists() else state_dir
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["type"] for line in lines][0] == "session.started"
    assert [line["type"] for line in lines][-1] == "session.completed"
    assert all(line["session_id"] == session_id for line in lines)
    assert not any("text" in line or "content" in line for line in lines)


def test_no_log_and_the_config_key_both_turn_it_off(meeting_ws, state_dir):
    """AC-4: --no-log writes nothing; so does logging.events false; --no-log wins over true."""
    assert run_cli(["-p", "x", "--no-log"], meeting_ws).returncode == 0
    assert not state_dir.exists()
    (meeting_ws / "kennel.json").write_text(json.dumps({"logging": {"events": False}}))
    assert run_cli(["-p", "x"], meeting_ws).returncode == 0
    assert not state_dir.exists()
    (meeting_ws / "kennel.json").write_text(json.dumps({"logging": {"events": True}}))
    assert run_cli(["-p", "x", "--no-log"], meeting_ws).returncode == 0
    assert not state_dir.exists()
    assert run_cli(["-p", "x"], meeting_ws).returncode == 0
    assert list((state_dir / "sessions").glob("*.jsonl"))  # the config key alone still logs


def test_status_shows_where_the_log_is(meeting_ws, state_dir):
    """AC-5: /status carries the log path."""
    p = run_cli([], meeting_ws, script={"turns": []}, stdin="/status\n/exit\n")
    assert p.returncode == 0, p.stderr
    assert f"log: {state_dir / 'sessions'}" in p.stdout
    assert p.stdout.count("log: ") == 1 and ".jsonl" in p.stdout


FAILING_TURN = {
    "turns": [
        [
            {"tool": "read", "arguments": {"path": "transcripts/2026-09-16.txt"}},
            {"error": "Foundation Models error: Generation error (status: 255): None", "status": 255},
        ],
        "recovered",
    ]
}


def test_one_shot_error_line_reports_what_the_turn_sent(meeting_ws):
    p = run_cli(["-p", "summarize"], meeting_ws, script=FAILING_TURN)
    assert p.returncode == 1, p.stderr
    assert "● Read transcripts/2026-09-16.txt" in p.stdout
    assert "Generation error (status: 255)" in p.stderr
    assert "bytes of tool output" in p.stderr
    assert "tokens (estimated) before the request." in p.stderr


def test_interactive_reports_the_failure_then_keeps_going(meeting_ws):
    p = run_cli([], meeting_ws, script=FAILING_TURN, stdin="summarize\n/usage\nagain\n/exit\n")
    assert p.returncode == 0, p.stderr
    assert "bytes of tool output" in p.stderr
    assert "recovered" in p.stdout  # the session survived the failed turn
    assert "turns: 1\ncompactions: 1" in p.stdout  # the failed turn is in the history


def test_interactive_session(meeting_ws):
    script = {"turns": [[{"tool": "glob", "arguments": {"pattern": "*.md"}}, {"text": "answer one"}], "answer two"]}
    stdin = "/help\n/status\nfirst\n/clear\nsecond\n/unknown\n/exit\n"
    p = run_cli([], meeting_ws, script=script, stdin=stdin)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert out.startswith("Kennel v0.1.0\nworkspace: ")
    assert "model: MockModel\nmode: local\npermissions: default\ntools: glob, grep, read, write (ask), edit (ask), shell (ask)" in out
    assert "/clear" in out and "forget the conversation" in out
    assert "/compact" in out and "/permissions" in out  # new commands are documented
    assert "turns: 0" in out and "permissions: edit=ask" in out
    assert "● Glob *.md" in out and "answer one" in out
    assert "(conversation cleared)" in out and "answer two" in out
    assert "unknown command /unknown" in out


def test_interactive_compact_command(meeting_ws):
    script = {"turns": ["one", "two", "three"]}
    stdin = "q1\nq2\n/compact\nq3\n/exit\n"
    p = run_cli([], meeting_ws, script=script, stdin=stdin)
    assert p.returncode == 0, p.stderr
    assert "(conversation compacted: 2 turns -> summary)" in p.stdout
    assert "three" in p.stdout


def test_interactive_compact_noop_when_empty(meeting_ws):
    p = run_cli([], meeting_ws, script={"turns": []}, stdin="/compact\n/exit\n")
    assert p.returncode == 0
    assert "(nothing to compact)" in p.stdout


def test_interactive_permissions_table_and_change(meeting_ws):
    script = {"turns": [[{"tool": "shell", "arguments": {"command": "echo hi"}}, {"text": "done"}]]}
    stdin = "/permissions\n/permissions shell allow\nrun it\n/exit\n"
    p = run_cli([], meeting_ws, script=script, stdin=stdin)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "tool" in out and "decision" in out and "session-grant" in out
    assert f"{'shell':<8}{'ask':<10}" in out  # before the change, from the first /permissions
    assert "(shell: ask -> allow for this session)" in out
    assert "● Shell echo hi" in out and "done" in out  # ran instead of being denied


def test_interactive_permissions_unknown_tool_and_bad_usage(meeting_ws):
    stdin = "/permissions nosuch allow\n/permissions shell\n/exit\n"
    p = run_cli([], meeting_ws, script={"turns": []}, stdin=stdin)
    assert p.returncode == 0, p.stderr
    assert "unknown tool 'nosuch'" in p.stdout
    assert "usage: /permissions" in p.stdout


def test_interactive_eof_exits_cleanly(meeting_ws):
    p = run_cli([], meeting_ws, script={"turns": []}, stdin="")
    assert p.returncode == 0 and p.stdout.startswith("Kennel")


def _run_doctor(args, ws: Path, home: Path):
    e = {**os.environ, "NO_COLOR": "1", "HOME": str(home)}
    proc = subprocess.run(
        [sys.executable, "-m", "kennel.cli.main", "doctor", *args],
        cwd=ws,
        capture_output=True,
        text=True,
        env=e,
        timeout=60,
    )
    return proc


def test_doctor_mock_provider_exits_0(meeting_ws, tmp_path):
    p = _run_doctor(["--provider", "mock"], meeting_ws, tmp_path / "home")
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith("Kennel ")
    assert "✓ Python" in p.stdout
    assert "effective: tools=" in p.stdout


def test_doctor_json_is_parseable(meeting_ws, tmp_path):
    p = _run_doctor(["--provider", "mock", "--json"], meeting_ws, tmp_path / "home")
    assert p.returncode == 0, p.stderr
    payload = json.loads(p.stdout)
    assert payload["ok"] is True and "checks" in payload


def test_doctor_invalid_project_config_exits_1_but_continues(meeting_ws, tmp_path):
    (meeting_ws / "kennel.json").write_text("{not valid json")
    p = _run_doctor(["--provider", "mock"], meeting_ws, tmp_path / "home")
    assert p.returncode == 1
    assert "✗" in p.stdout and "✓ Python" in p.stdout  # other checks still ran


def test_doctor_does_not_break_existing_parsing(meeting_ws):
    """Adding the `doctor` subcommand must not regress how `.`, a path, or `-p` parse (AC-4)."""

    def run(argv, stdin=""):
        e = {**os.environ, "NO_COLOR": "1", "KENNEL_MOCK_SCRIPT": json.dumps({"turns": ["ok"]})}
        return subprocess.run(
            [sys.executable, "-m", "kennel.cli.main", *argv, "--provider", "mock"],
            cwd=meeting_ws,
            input=stdin,
            capture_output=True,
            text=True,
            env=e,
            timeout=60,
        )

    assert run([".", "-p", "x"]).returncode == 0
    assert run([str(meeting_ws), "-p", "hi"]).returncode == 0
    assert run(["-p", "hi"], stdin="").returncode == 0


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


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty needed")
def test_interactive_ctrl_c_cancels_the_turn(meeting_ws):
    """Ctrl-C while answering prints (cancelled) and returns to a usable prompt."""
    import pty
    import select
    import time

    script = {"turns": [[{"sleep": 30}, {"text": "never"}], "after cancel"]}
    env = {**os.environ, "NO_COLOR": "1", "KENNEL_MOCK_SCRIPT": json.dumps(script)}
    pid, fd = pty.fork()
    if pid == 0:  # child
        os.chdir(meeting_ws)
        os.execvpe(sys.executable, [sys.executable, "-m", "kennel.cli.main", ".", "--provider", "mock"], env)
    output = b""
    deadline = time.time() + 60
    asked = interrupted = asked_again = quit_sent = False
    try:
        while time.time() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
            # Always wait for the prompt itself, never for the line before it: input typed
            # before a prompt is printed is discarded (#34). Only the Ctrl-C below is meant
            # to arrive mid-turn.
            if not asked and output.endswith(b"> "):
                os.write(fd, b"slow question\n")
                asked = True
            elif asked and not interrupted and b"slow question" in output:
                time.sleep(0.5)  # let the turn reach the sleeping provider call
                os.write(fd, b"\x03")  # Ctrl-C: SIGINT to the foreground process group
                interrupted = True
            elif interrupted and not asked_again and b"(cancelled)" in output and output.endswith(b"> "):
                os.write(fd, b"second question\n")
                asked_again = True
            elif asked_again and not quit_sent and b"after cancel" in output and output.endswith(b"> "):
                os.write(fd, b"/exit\n")
                quit_sent = True
            if os.waitpid(pid, os.WNOHANG)[1]:
                break
    finally:
        os.close(fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    text = output.decode(errors="replace")
    assert "(cancelled)" in text, text
    assert "after cancel" in text, text  # the session is still usable after the interrupt
    assert "Traceback" not in text and "never" not in text

def test_usage_command_reports_the_context_window(meeting_ws):
    script = {"turns": [[{"tool": "read", "arguments": {"path": "transcripts/2026-09-16.txt"}}, {"text": "summarized"}]]}
    stdin = "/usage\nsummarize it\n/usage\n/status\n/exit\n"
    p = run_cli([], meeting_ws, script=script, stdin=stdin)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "/usage    show how full the model's context window is" not in out  # /help was not asked for
    blocks = out.split("window: ")
    assert len(blocks) == 3, out  # one per /usage
    for block in blocks[1:]:
        assert block.startswith("4096 tokens\nused: ")
        assert "(estimated)\ncontext: " in block
    before = int(blocks[1].split("used: ")[1].split(" tokens")[0])
    after = int(blocks[2].split("used: ")[1].split(" tokens")[0])
    assert after > before  # the turn and its tool output now sit in the window
    assert "turns: 0\ncompactions: 0" in blocks[1] and "turns: 1\ncompactions: 0" in blocks[2]
    assert "context: " in out.split("session_id: ")[-1]  # /status carries a context line


def _drive_pty(ws: Path, script, steps, timeout=60.0) -> str:
    """Run the interactive CLI under a pty and play *steps* against its output.

    Each step is ``(predicate, payload)``: once ``predicate(output_so_far)`` holds, the
    payload is delivered once -- bytes are written to the tty, a callable is handed the fd
    so it can time its own keystrokes. Reading continues until the child exits or *timeout*.
    """
    import pty
    import select
    import time

    env = {**os.environ, "NO_COLOR": "1", "KENNEL_MOCK_SCRIPT": json.dumps(script)}
    pid, fd = pty.fork()
    if pid == 0:  # child
        os.chdir(ws)
        os.execvpe(sys.executable, [sys.executable, "-m", "kennel.cli.main", ".", "--provider", "mock"], env)
    output = b""
    pending = list(steps)
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
            if pending and pending[0][0](output):
                _, payload = pending.pop(0)
                if callable(payload):
                    payload(fd)
                else:
                    os.write(fd, payload)
            if os.waitpid(pid, os.WNOHANG)[1]:
                break
    finally:
        os.close(fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    assert not pending, f"never reached step {len(steps) - len(pending)}:\n{output.decode(errors='replace')}"
    return output.decode(errors="replace")


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty needed")
def test_interactive_discards_input_typed_during_a_turn(meeting_ws):
    """Enters and a half-typed line entered mid-turn never become the next turn."""
    import time

    def type_ahead(fd):
        time.sleep(0.5)  # the turn is under way; three bare Enters and a partial line
        for _ in range(3):
            os.write(fd, b"\n")
            time.sleep(0.15)
        os.write(fd, b"half-typed")

    script = {"turns": [[{"sleep": 3}, {"text": "answer one"}], "answer two"]}
    text = _drive_pty(
        meeting_ws,
        script,
        [
            (lambda o: o.endswith(b"> "), b"first question\n"),
            (lambda o: b"first question" in o, type_ahead),
            (lambda o: b"answer one" in o and o.endswith(b"> "), b"/status\n"),
            (lambda o: b"permission_mode:" in o and o.endswith(b"> "), b"/exit\n"),
        ],
    )
    # One prompt after the turn, not one per swallowed Enter: input() writes the prompt
    # without a newline, so stacked prompts show up as "> > " on a single line.
    assert "> > " not in text, text
    assert text.count("answer one") == 1, text
    # "/status" arrived on its own line: had "half-typed" survived, "half-typed/status"
    # would have been a prompt, not a command, and the mock would have answered it.
    assert "permission_mode:" in text, text
    assert "answer two" not in text, text
    assert "Traceback" not in text, text


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty needed")
def test_typed_ahead_does_not_answer_the_permission_prompt(meeting_ws):
    """A "y" typed before the Allow prompt appears must not approve the write."""
    import time

    def type_ahead_yes(fd):
        time.sleep(0.8)  # well before the prompt: the turn sleeps for 3 seconds first
        os.write(fd, b"y\n")

    script = {
        "turns": [
            [{"sleep": 3}, {"tool": "write", "arguments": {"path": "note.txt", "content": "by the mock"}}, {"text": "wrote the file"}],
            "second answer",
        ]
    }
    text = _drive_pty(
        meeting_ws,
        script,
        [
            (lambda o: o.endswith(b"> "), b"please write the note\n"),
            (lambda o: b"please write the note" in o, type_ahead_yes),
            (lambda o: b"[y] once" in o, b"n\n"),  # the answer the user actually saw
            (lambda o: b"wrote the file" in o and o.endswith(b"> "), b"/exit\n"),
        ],
    )
    assert "[y] once" in text, text
    assert "⊘ denied: Write note.txt" in text, text
    assert not (meeting_ws / "note.txt").exists(), text
    assert "Traceback" not in text, text


def test_discard_typed_ahead_is_a_noop_without_a_tty(meeting_ws):
    """Off a tty the helper touches nothing, so piped stdin keeps working."""
    import io

    from kennel.cli.terminal import discard_typed_ahead

    queued = io.StringIO("queued\n")
    discard_typed_ahead(queued)
    assert queued.read() == "queued\n"  # a StringIO is not a tty: left alone

    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd) as pipe, os.fdopen(write_fd, "w") as sink:
        sink.write("piped\n")
        discard_typed_ahead(pipe)  # a real fd, but not a terminal

    closed = io.StringIO()
    closed.close()
    discard_typed_ahead(closed)
    discard_typed_ahead()  # under pytest stdin is not a tty either

    p = run_cli([], meeting_ws, script={"turns": ["only answer"]}, stdin="hello\n/exit\n")
    assert p.returncode == 0, p.stderr
    assert "only answer" in p.stdout and "Traceback" not in p.stderr
