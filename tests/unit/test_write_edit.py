import os

import pytest

from kennel.errors import ToolArgumentError, ToolExecutionError, WorkspaceEscapeError
from kennel.tools.edit import EditTool
from kennel.tools.write import WriteTool


async def write(ctx, **args):
    return await WriteTool().execute(WriteTool().validate(args), ctx)


async def edit(ctx, **args):
    return await EditTool().execute(EditTool().validate(args), ctx)


async def test_write_creates_with_parents(ctx, ws_dir):
    r = await write(ctx, path="new/dir/file.md", content="# Hi\n")
    assert (ws_dir / "new/dir/file.md").read_text() == "# Hi\n"
    assert r.content == "Created new/dir/file.md (5 bytes)"
    assert r.metadata == {"path": "new/dir/file.md", "bytes": 5, "overwritten": False}
    assert not [p for p in (ws_dir / "new/dir").iterdir() if p.name.endswith(".tmp")]


async def test_write_overwrite_preserves_mode(ctx, ws_dir):
    target = ws_dir / "a.txt"
    os.chmod(target, 0o640)
    r = await write(ctx, path="a.txt", content="new\n")
    assert target.read_text() == "new\n" and r.metadata["overwritten"] is True
    assert oct(target.stat().st_mode & 0o777) == "0o640"


async def test_write_errors(ctx):
    with pytest.raises(ToolExecutionError, match="directory"):
        await write(ctx, path="docs", content="x")
    with pytest.raises(WorkspaceEscapeError):
        await write(ctx, path="../outside/evil.txt", content="x")
    with pytest.raises(WorkspaceEscapeError):
        await write(ctx, path="dir_out/evil.txt", content="x")


def test_write_permission_details(ctx):
    tool = WriteTool()
    d = tool.permission_details({"path": "brand_new.txt", "content": "a\nb\n"}, ctx)
    assert d.startswith("Creates new file brand_new.txt (4 bytes, 2 lines)")
    d = tool.permission_details({"path": "a.txt", "content": "alpha\nBETA\ngamma\n"}, ctx)
    assert "Overwrites existing file a.txt" in d and "-beta" in d and "+BETA" in d


async def test_edit_unique_replacement(ctx, ws_dir):
    r = await edit(ctx, path="a.txt", old_text="beta", new_text="delta\nepsilon")
    assert (ws_dir / "a.txt").read_text() == "alpha\ndelta\nepsilon\ngamma\n"
    assert r.content == "Edited a.txt (1 replacement, +1 lines)"


async def test_edit_failures(ctx):
    with pytest.raises(ToolExecutionError, match="not found"):
        await edit(ctx, path="a.txt", old_text="zeta", new_text="x")
    with pytest.raises(ToolExecutionError, match="ambiguous"):
        await edit(ctx, path="docs/b.md", old_text="hello", new_text="x")
    with pytest.raises(ToolExecutionError, match="No such file"):
        await edit(ctx, path="nope.txt", old_text="a", new_text="b")
    with pytest.raises(ToolArgumentError):
        await edit(ctx, path="a.txt", old_text="", new_text="b")


def test_edit_permission_details_is_diff(ctx):
    d = EditTool().permission_details({"path": "a.txt", "old_text": "beta", "new_text": "BETA"}, ctx)
    assert d.startswith("--- a/a.txt") and "-beta" in d and "+BETA" in d
