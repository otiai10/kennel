import shutil

import pytest

from kennel.errors import ToolArgumentError
from kennel.tools.grep import GrepTool


async def run(ctx, use_rg=False, **args):
    tool = GrepTool(use_rg=use_rg)
    return await tool.execute(tool.validate(args), ctx)


async def test_basic_python(ctx):
    r = await run(ctx, pattern="hello")
    lines = r.content.splitlines()
    assert "docs/b.md:3: hello World" in lines
    assert "docs/b.md:4: hello again" in lines
    assert "src/main.py:2: print('Hello')" in lines  # case-insensitive default
    assert not any("node_modules" in line or "hidden" in line or "outside" in line for line in lines)
    assert r.metadata["engine"] == "python" and not r.truncated


async def test_case_sensitive_and_glob(ctx):
    r = await run(ctx, pattern="hello", case_sensitive=True)
    assert "src/main.py" not in r.content
    r = await run(ctx, pattern="hello", glob="*.py")
    assert r.content.splitlines() == ["src/main.py:2: print('Hello')"]


async def test_no_match_and_limits(ctx):
    r = await run(ctx, pattern="zzzz")
    assert r.content == "No matches for 'zzzz'."
    r = await run(ctx, pattern="hello", max_results=1)
    assert len([line for line in r.content.splitlines() if ":" in line and not line.startswith("...")]) == 1
    assert r.truncated and "more matches not shown" in r.content


async def test_invalid_regex(ctx):
    with pytest.raises(ToolArgumentError):
        await run(ctx, pattern="(unclosed")


async def test_single_file_and_snippet_bound(ctx, ws_dir):
    (ws_dir / "long.txt").write_text("x" * 1000 + "needle" + "y" * 1000 + "\n")
    r = await run(ctx, pattern="needle", path="long.txt")
    line = r.content.splitlines()[0]
    assert line.startswith("long.txt:1: ")
    assert len(line) < 300


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
async def test_rg_parity(ctx):
    py = await run(ctx, pattern="hello", glob="*.md")
    rg = await run(ctx, use_rg=True, pattern="hello", glob="*.md")
    assert rg.metadata["engine"] == "rg"
    assert py.content.splitlines() == rg.content.splitlines()
    with pytest.raises(ToolArgumentError):
        await run(ctx, use_rg=True, pattern="(unclosed")
