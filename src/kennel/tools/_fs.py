"""Filesystem helpers shared by the file tools."""

from __future__ import annotations

import difflib
import os
import re
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


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob with ``**`` support into a regex over POSIX relative paths."""
    i, n = 0, len(pattern)
    out: list[str] = []
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1 : j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def matches_glob(pattern: str, relative_posix: str) -> bool:
    """Match a workspace-relative POSIX path.

    Patterns without ``/`` match the basename at any depth, or the whole path
    with ``*`` allowed to cross directories (so ``*transcript*`` finds
    ``transcripts/a.txt``).
    """
    if "/" not in pattern:
        if glob_to_regex(pattern).match(relative_posix.rsplit("/", 1)[-1]):
            return True
        loose = re.compile("^" + glob_to_regex(pattern).pattern[1:-1].replace("[^/]*", ".*").replace("[^/]", ".") + "$")
        return loose.match(relative_posix) is not None
    if pattern.startswith("./"):
        pattern = pattern[2:]
    return glob_to_regex(pattern).match(relative_posix) is not None
