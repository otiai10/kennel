"""Workspace boundary and the shared path resolver.

Every file tool must go through :meth:`Workspace.resolve`; no tool implements
its own boundary check. The check is done on real paths (symlinks resolved), so
symlinks pointing outside the workspace are rejected, and a destination whose
leaf does not exist yet is validated through its nearest existing ancestor.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import WorkspaceError, WorkspaceEscapeError

DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".idea",
        ".vscode",
        ".cache",
        "dist",
        "build",
        ".eggs",
    }
)


class Workspace:
    """A directory that bounds every filesystem operation of an agent."""

    def __init__(self, root: str | os.PathLike[str] = ".", *, ignored_dirs: frozenset[str] | None = None):
        path = Path(root).expanduser()
        if not path.exists():
            raise WorkspaceError(f"Workspace does not exist: {path}")
        if not path.is_dir():
            raise WorkspaceError(f"Workspace is not a directory: {path}")
        # Real path: if the root itself is a symlink, the target is the boundary.
        self.root: Path = path.resolve(strict=True)
        self.ignored_dirs: frozenset[str] = DEFAULT_IGNORED_DIRS if ignored_dirs is None else ignored_dirs

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Workspace({str(self.root)!r})"

    # -- path resolution -------------------------------------------------

    def resolve(self, user_path: str | os.PathLike[str]) -> Path:
        """Resolve ``user_path`` (relative to the root, or absolute) to a real path inside the workspace.

        Raises :class:`WorkspaceEscapeError` if the real path is outside the root,
        including escapes through ``..``, absolute paths, or symlinks anywhere in
        the path. The leaf may not exist yet (write destinations); in that case
        the nearest existing ancestor is resolved and the remaining components
        must not contain ``..``.
        """
        raw = os.fspath(user_path).strip()
        if raw == "":
            raise WorkspaceError("Path must not be empty")
        if "\x00" in raw:
            raise WorkspaceError("Path must not contain NUL")
        p = Path(raw).expanduser()
        candidate = p if p.is_absolute() else self.root / p

        # Split into the deepest existing ancestor and the remaining (nonexistent) tail.
        existing = candidate
        tail: list[str] = []
        while not existing.exists():
            if existing.parent == existing:  # filesystem root
                break
            tail.insert(0, existing.name)
            existing = existing.parent
        if any(part in ("..", ".") for part in tail):
            raise WorkspaceEscapeError(f"Path escapes the workspace: {raw}")

        real_base = existing.resolve()
        final = real_base.joinpath(*tail) if tail else real_base
        if final != self.root and not self._is_within_root(final):
            raise WorkspaceEscapeError(f"Path escapes the workspace: {raw}")
        return final

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False

    def contains(self, path: Path) -> bool:
        """True if an already-resolved real path is inside the workspace."""
        return path == self.root or self._is_within_root(path)

    def relative(self, path: Path) -> str:
        """Workspace-relative display form of a resolved path ('.' for the root)."""
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return str(path)
        return "." if str(rel) == "." else rel.as_posix()

    def is_ignored_dir(self, name: str) -> bool:
        return name in self.ignored_dirs


def count_files(workspace: Workspace, directory: Path, limit: int = 200) -> int:
    """Count regular files under ``directory`` (ignored/hidden entries skipped), stopping at ``limit``."""
    count = 0
    for _dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
        dirnames[:] = [d for d in dirnames if not workspace.is_ignored_dir(d) and not d.startswith(".")]
        count += sum(1 for name in filenames if not name.startswith("."))
        if count >= limit:
            return limit
    return count


def workspace_overview(workspace: Workspace, max_entries: int = 20, count_limit: int = 200) -> str:
    """One line listing the top-level entries (directories first), so a small model starts from real names.

    Example: ``Top-level entries: docs/ (3 files), src/ (200+ files), README.md``.
    """
    entries: list[str] = []
    try:
        children = sorted(workspace.root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except OSError:
        return "Top-level entries: (unreadable)"
    for child in children:
        if child.name.startswith(".") or workspace.is_ignored_dir(child.name):
            continue
        if child.is_dir():
            if child.is_symlink():
                continue
            n = count_files(workspace, child, count_limit)
            shown = f"{count_limit}+" if n >= count_limit else str(n)
            entries.append(f"{child.name}/ ({shown} files)")
        elif child.is_file():
            entries.append(child.name)
        if len(entries) >= max_entries:
            entries.append("...")
            break
    return "Top-level entries: " + (", ".join(entries) if entries else "(empty)")
