import os
from pathlib import Path

import pytest

from kennel.permissions import PermissionManager
from kennel.tools.base import ToolContext, ToolLimits
from kennel.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def ws_dir(tmp_path: Path) -> Path:
    """A workspace with text, binary, hidden, ignored and symlinked entries plus an outside sibling."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("alpha\nbeta\ngamma\n")
    (root / "docs").mkdir()
    (root / "docs" / "b.md").write_text("# Title\n\nhello World\nhello again\n")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    print('Hello')\n")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.txt").write_text("hidden hello\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("hello from node_modules\n")
    (root / "bin.dat").write_bytes(b"\x00\x01hello\x02")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside hello\n")
    os.symlink(outside / "secret.txt", root / "link_out")
    os.symlink(outside, root / "dir_out")
    os.symlink(root / "a.txt", root / "link_in")
    return root


@pytest.fixture
def ws(ws_dir: Path) -> Workspace:
    return Workspace(ws_dir)


@pytest.fixture
def limits() -> ToolLimits:
    return ToolLimits(max_output_bytes=4096, max_read_lines=50, glob_max_results=100, grep_max_results=50)


@pytest.fixture
def ctx(ws: Workspace, limits: ToolLimits) -> ToolContext:
    return ToolContext(workspace=ws, permission_manager=PermissionManager(), limits=limits)


@pytest.fixture
def meeting_ws(tmp_path: Path) -> Path:
    """Copy of the meeting fixture so tests can mutate it."""
    import shutil

    dst = tmp_path / "meeting_project"
    shutil.copytree(FIXTURES / "meeting_project", dst)
    return dst
