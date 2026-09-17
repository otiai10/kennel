import pytest

from kennel.errors import ToolArgumentError, ToolExecutionError
from kennel.tools.read import ReadTool


async def run(ctx, **args):
    return await ReadTool().execute(ReadTool().validate(args), ctx)


async def test_numbered_lines(ctx):
    r = await run(ctx, path="a.txt")
    assert r.content == "     1| alpha\n     2| beta\n     3| gamma"
    assert r.metadata["total_lines"] == 3 and not r.truncated


async def test_range(ctx):
    r = await run(ctx, path="a.txt", start_line=2, end_line=2)
    assert r.content.startswith("     2| beta")
    assert "showing lines 2-2 of 3" in r.content and r.truncated
    assert r.metadata["end_line"] == 2


async def test_clamped_to_limit(ctx, ws_dir):
    (ws_dir / "big.txt").write_text("".join(f"line {i}\n" for i in range(1, 201)))
    r = await run(ctx, path="big.txt")
    assert r.content.count("\n") == 50  # 50 lines + note
    assert "call read again with start_line=51" in r.content
    assert r.metadata["clamped"] is False  # nothing explicit was requested
    r2 = await run(ctx, path="big.txt", start_line=51, end_line=1000)
    assert r2.content.startswith("    51| line 51") and r2.metadata["clamped"] is True
    assert r2.metadata["end_line"] == 100


async def test_errors(ctx):
    with pytest.raises(ToolExecutionError, match="No such file"):
        await run(ctx, path="missing.txt")
    with pytest.raises(ToolExecutionError, match="directory"):
        await run(ctx, path="docs")
    with pytest.raises(ToolExecutionError, match="binary"):
        await run(ctx, path="bin.dat")
    with pytest.raises(ToolArgumentError):
        await run(ctx, path="a.txt", start_line=0)
    with pytest.raises(ToolArgumentError):
        await run(ctx, path="a.txt", start_line=3, end_line=2)
    with pytest.raises(ToolArgumentError, match="past the end"):
        await run(ctx, path="a.txt", start_line=10)


async def test_empty_and_escape(ctx, ws_dir):
    (ws_dir / "empty.txt").write_text("")
    r = await run(ctx, path="empty.txt")
    assert "is empty" in r.content
    from kennel.errors import WorkspaceEscapeError

    with pytest.raises(WorkspaceEscapeError):
        await run(ctx, path="link_out")
