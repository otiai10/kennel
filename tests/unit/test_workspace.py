import os
from pathlib import Path

import pytest

from kennel.errors import WorkspaceError, WorkspaceEscapeError
from kennel.workspace import Workspace


def test_root_is_real_path(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    ws = Workspace(link)
    assert ws.root == real.resolve()
    assert ws.resolve("x.txt") == real.resolve() / "x.txt"


def test_missing_or_file_root(tmp_path: Path):
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path / "nope")
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(WorkspaceError):
        Workspace(f)


def test_resolve_relative_and_absolute_inside(ws: Workspace):
    assert ws.resolve("a.txt") == ws.root / "a.txt"
    assert ws.resolve("./docs/b.md") == ws.root / "docs" / "b.md"
    assert ws.resolve(str(ws.root / "src" / "main.py")) == ws.root / "src" / "main.py"
    assert ws.resolve(".") == ws.root
    assert ws.resolve("docs/../a.txt") == ws.root / "a.txt"


@pytest.mark.parametrize("bad", ["../outside/secret.txt", "docs/../../outside/secret.txt", "/etc/passwd", "~/whatever"])
def test_resolve_rejects_escapes(ws: Workspace, bad: str):
    with pytest.raises(WorkspaceEscapeError):
        ws.resolve(bad)


def test_resolve_rejects_symlink_escapes(ws: Workspace):
    with pytest.raises(WorkspaceEscapeError):
        ws.resolve("link_out")
    with pytest.raises(WorkspaceEscapeError):
        ws.resolve("dir_out/secret.txt")
    with pytest.raises(WorkspaceEscapeError):
        ws.resolve("dir_out/new_file.txt")  # nonexistent leaf under symlinked parent


def test_resolve_allows_symlink_inside(ws: Workspace):
    assert ws.resolve("link_in") == ws.root / "a.txt"


def test_resolve_nonexistent_leaf(ws: Workspace):
    assert ws.resolve("new/dir/file.md") == ws.root / "new" / "dir" / "file.md"
    with pytest.raises(WorkspaceEscapeError):
        ws.resolve("new/../../escape.md")


def test_resolve_empty_and_nul(ws: Workspace):
    with pytest.raises(WorkspaceError):
        ws.resolve("")
    with pytest.raises(WorkspaceError):
        ws.resolve("a\x00b")


def test_relative_and_contains(ws: Workspace):
    assert ws.relative(ws.root) == "."
    assert ws.relative(ws.root / "docs" / "b.md") == "docs/b.md"
    assert ws.contains(ws.root / "docs")
    assert not ws.contains(ws.root.parent / "outside")


def test_workspace_overview(ws: Workspace, tmp_path: Path):
    from kennel.workspace import workspace_overview

    text = workspace_overview(ws)
    assert text.startswith("Top-level entries: ")
    assert "docs/ (1 files)" in text and "src/ (1 files)" in text and "a.txt" in text
    assert "node_modules" not in text and ".hidden" not in text and "dir_out" not in text
    empty = tmp_path / "empty"
    empty.mkdir()
    assert workspace_overview(Workspace(empty)) == "Top-level entries: (empty)"
    big = tmp_path / "big"
    (big / "pkg").mkdir(parents=True)
    for i in range(250):
        (big / "pkg" / f"m{i}.py").write_text("")
    for i in range(25):
        (big / f"f{i:02d}.txt").write_text("")
    text = workspace_overview(Workspace(big), max_entries=5, count_limit=200)
    assert text.endswith(", ...") and text.count(",") == 5
    assert text.startswith("Top-level entries: pkg/ (200+ files), f00.txt")  # directories first
