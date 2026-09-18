"""Filesystem helpers shared by the file tools."""

from __future__ import annotations

import difflib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from ..workspace import Workspace


def is_probably_binary(path: Path, sample: int = 8192) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(sample)
    except OSError:
        return True
    return b"\x00" in chunk


def atomic_write_text(path: Path, text: str) -> int:
    """Write ``text`` to ``path`` via a temp file + rename. Returns bytes written."""
    data = text.encode("utf-8")
    mode = None
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        pass
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(data)


def unified_diff(old: str, new: str, display_path: str, max_lines: int = 200) -> str:
    lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{display_path}",
            tofile=f"b/{display_path}",
        )
    )
    text = "".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    if len(lines) > max_lines:
        text = "".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more diff lines)\n"
    return text


def iter_files(workspace: Workspace, base: Path, *, include_hidden: bool = False) -> Iterator[Path]:
    """Yield real file paths under ``base`` in deterministic order.

    Ignored directories are skipped, hidden entries are skipped unless
    ``include_hidden``, symlinked directories are never followed and symlinked
    files are only yielded when their target stays inside the workspace.
    """
    if base.is_file():
        yield base
        return
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not workspace.is_ignored_dir(d) and (include_hidden or not d.startswith("."))
        )
        for name in sorted(filenames):
            if not include_hidden and name.startswith("."):
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                try:
                    real = path.resolve(strict=True)
                except OSError:
                    continue
                if not workspace.contains(real) or not real.is_file():
                    continue
            yield path
