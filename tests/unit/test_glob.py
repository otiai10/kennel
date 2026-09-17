import os

import pytest

from kennel.rules import matches_glob
from kennel.tools.glob import GlobTool


async def run(ctx, **args):
    return await GlobTool().execute(GlobTool().validate(args), ctx)


@pytest.mark.parametrize(
    "pattern,path,expected",
    [
        ("**/*.md", "docs/b.md", True),
        ("**/*.md", "b.md", True),
        ("*.md", "docs/b.md", True),
        ("docs/*.md", "docs/b.md", True),
        ("docs/*.md", "docs/sub/b.md", False),
        ("*.py", "docs/b.md", False),
        ("*transcript*", "transcripts/2026.txt", True),
        ("src/**", "src/a/b.py", True),
        ("?.txt", "a.txt", True),
        ("[ab].txt", "b.txt", True),
    ],
)
def test_matches_glob(pattern, path, expected):
    assert matches_glob(pattern, path) is expected


async def test_recursive_and_basename_patterns(ctx):
    r = await run(ctx, pattern="**/*.md")
    assert r.content == "docs/b.md"
    r = await run(ctx, pattern="*.txt")
    assert r.content.splitlines() == ["a.txt"]  # hidden and outside-symlinked files excluded
    assert r.metadata == {"total": 1, "returned": 1, "pattern": "*.txt"}
    r = await run(ctx, pattern="link_*")
    assert r.content == "link_in"  # symlink whose target is inside the workspace


async def test_ignores_ignored_dirs_and_hidden(ctx):
    r = await run(ctx, pattern="**/*.js")
    assert r.content.startswith("No files matched")
    r = await run(ctx, pattern="**/secret.txt")
    assert "No files matched" in r.content
    r = await run(ctx, pattern="**/secret.txt", include_hidden=True)
    assert r.content == ".hidden/secret.txt"


async def test_symlink_escape_not_listed(ctx):
    r = await run(ctx, pattern="**/*")
    assert "link_out" not in r.content
    assert "dir_out" not in r.content


async def test_no_match_gives_overview(ctx):
    r = await run(ctx, pattern="*.rs")
    assert "Top-level entries here:" in r.content
    assert "docs/ (1 files)" in r.content
    assert "node_modules" not in r.content


async def test_path_argument_and_escape(ctx):
    r = await run(ctx, pattern="*.md", path="docs")
    assert r.content == "docs/b.md"
    from kennel.errors import WorkspaceEscapeError

    with pytest.raises(WorkspaceEscapeError):
        await run(ctx, pattern="*", path="../outside")


async def test_truncation_is_deterministic(ctx, ws_dir):
    for i in range(150):
        (ws_dir / f"f{i:03d}.log").write_text("x")
    ctx.limits.glob_max_results = 10
    r = await run(ctx, pattern="*.log")
    lines = r.content.splitlines()
    assert lines[:10] == [f"f{i:03d}.log" for i in range(10)]
    assert r.truncated and r.metadata["total"] == 150 and r.metadata["returned"] == 10
    assert "140 more matches" in lines[-1]
    assert os.path.basename(lines[0]) == "f000.log"
