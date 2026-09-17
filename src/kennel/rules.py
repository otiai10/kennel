"""Glob matching and permission rule syntax.

Two things live here because they must agree: the glob translation used by the
``glob`` tool to find files, and the one used by permission rules to decide
whether ``write(docs/**)`` covers a call. The module imports nothing from
``kennel.tools`` so that :mod:`kennel.permissions` can use it without a cycle.

A rule key is ``name`` or ``name(specifier)``. The bare form applies to every
call of that tool; the specifier form applies only when the tool says the
arguments match it (:meth:`kennel.Tool.match_rule`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from .errors import ConfigurationError

_RULE = re.compile(r"^\s*(?P<name>[A-Za-z_][\w.-]*)\s*(?:\(\s*(?P<spec>.*?)\s*\)\s*)?$", re.DOTALL)


def parse_rule(key: str) -> tuple[str, str | None]:
    """Split a permission rule key into ``(tool_name, specifier | None)``.

    >>> parse_rule("shell(git *)")
    ('shell', 'git *')
    """
    match = _RULE.match(key)
    if match is None:
        raise ConfigurationError(
            f"Invalid permission rule {key!r}; expected 'tool' or 'tool(specifier)', e.g. 'shell(git *)'"
        )
    spec = match.group("spec")
    if spec is not None and not spec:
        raise ConfigurationError(f"Invalid permission rule {key!r}; the specifier must not be empty")
    return match.group("name"), spec


@lru_cache(maxsize=1024)
def glob_to_regex(pattern: str, cross_slash: bool = False) -> re.Pattern[str]:
    """Translate a glob with ``**`` support into a regex over POSIX relative paths.

    With ``cross_slash`` a single ``*`` and ``?`` also match ``/``, which is what
    free-form text (a command line) and the loose path fallback want.
    """
    any_run, any_one = (".*", ".") if cross_slash else ("[^/]*", "[^/]")
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
            out.append(any_run)
        elif c == "?":
            out.append(any_one)
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
    return re.compile("^" + "".join(out) + "$", re.DOTALL)


def matches_glob(pattern: str, relative_posix: str) -> bool:
    """Match a workspace-relative POSIX path.

    Patterns without ``/`` match the basename at any depth, or the whole path
    with ``*`` allowed to cross directories (so ``*transcript*`` finds
    ``transcripts/a.txt``).
    """
    if "/" not in pattern:
        if glob_to_regex(pattern).match(relative_posix.rsplit("/", 1)[-1]):
            return True
        return glob_to_regex(pattern, cross_slash=True).match(relative_posix) is not None
    if pattern.startswith("./"):
        pattern = pattern[2:]
    return glob_to_regex(pattern).match(relative_posix) is not None


def matches_text(pattern: str, text: str) -> bool:
    """Match a free-form string (a shell command line, a query) against a glob.

    Unlike :func:`matches_glob` there is no path structure, so ``*`` also
    crosses ``/`` and the whole string must match (``git *`` matches
    ``git status`` but not ``sudo git status``).
    """
    return glob_to_regex(pattern, cross_slash=True).match(text) is not None


def normalize_path(value: str) -> str:
    """Workspace-relative POSIX form of a tool's ``path`` argument, for rule matching.

    Lexical only, but ``..`` is collapsed: ``docs/../src/x`` becomes ``src/x``, so
    ``write(docs/**)`` cannot be satisfied by a path that leaves ``docs``.
    Absolute paths and paths that climb above the root keep their prefix (the
    tools resolve them through the workspace; a rule written with a relative
    glob simply will not match).
    """
    path = value.strip().replace("\\", "/")
    if not path:
        return "."
    absolute = path.startswith("/")
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == ".." and parts and parts[-1] != "..":
            parts.pop()
            continue
        if part == ".." and absolute:
            continue  # cannot climb above the root
        parts.append(part)
    out = "/".join(parts)
    return ("/" + out) if absolute else (out or ".")


def match_arguments(specifier: str, arguments: Mapping[str, Any]) -> bool:
    """Default rule matching for a tool that does not define its own.

    The argument values are joined with spaces in declaration order and matched
    as free-form text, so ``mytool(report *)`` covers
    ``{"action": "report", "id": "7"}``.
    """
    text = " ".join(str(value) for value in arguments.values())
    return matches_text(specifier, text)


def match_path_argument(
    specifier: str, arguments: Mapping[str, Any], *, key: str = "path", default: str = ""
) -> bool:
    """Rule matching for tools whose specifier is a path glob (``write(docs/**)``)."""
    value = arguments.get(key)
    path = str(value) if value not in (None, "") else default
    if not path:
        return False
    return matches_glob(specifier, normalize_path(path))
