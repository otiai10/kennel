import time

from kennel.tools.shell import ShellTool


async def run(ctx, **args):
    return await ShellTool().execute(ShellTool().validate(args), ctx)


async def test_runs_in_workspace(ctx, ws_dir):
    r = await run(ctx, command="pwd && echo out && echo err 1>&2 && exit 3")
    assert r.metadata["exit_code"] == 3 and not r.metadata["timed_out"]
    assert f"stdout:\n{ws_dir.resolve()}\nout" in r.content
    assert "stderr:\nerr" in r.content
    assert r.content.startswith("exit code: 3")


async def test_timeout_kills_process_group(ctx):
    start = time.monotonic()
    r = await run(ctx, command="sleep 5; echo late", timeout_seconds=1)
    assert time.monotonic() - start < 4
    assert r.metadata["timed_out"] and "killed after 1s timeout" in r.content
    assert "late" not in r.content


async def test_output_truncated(ctx):
    r = await run(ctx, command="yes | head -c 100000")
    assert r.truncated and "[truncated" in r.content
    assert len(r.content.encode()) < 8000


async def test_env_allowlist_and_no_stdin(ctx, monkeypatch):
    monkeypatch.setenv("KENNEL_TEST_SECRET", "s3cr3t")
    r = await run(ctx, command="echo \"[$KENNEL_TEST_SECRET]\"; cat")
    assert "[]" in r.content and "s3cr3t" not in r.content
    assert r.metadata["exit_code"] == 0  # cat exits immediately: stdin is /dev/null
    ctx.environment = {"KENNEL_EXTRA": "yes"}
    r = await run(ctx, command="echo $KENNEL_EXTRA")
    assert "yes" in r.content


def test_warnings_and_details(ctx):
    tool = ShellTool()
    assert tool.permission_warnings({"command": "sudo rm -rf /tmp/x"}, ctx) == (
        "recursive force delete (rm -rf)",
        "runs with sudo",
        "references paths outside the workspace",
    )
    assert tool.permission_warnings({"command": "git status"}, ctx) == ()
    assert "git reset --hard" in tool.permission_details({"command": "git reset --hard"}, ctx)
    assert str(ctx.workspace.root) in tool.permission_details({"command": "ls"}, ctx)
    assert "destructive git operation" in tool.permission_warnings({"command": "git reset --hard HEAD~1"}, ctx)
    assert tool.permission_warnings({"command": "echo x > ~/notes.txt"}, ctx) == ("references paths outside the workspace",)
