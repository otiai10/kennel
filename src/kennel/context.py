"""Context management helpers: output bounding, chunking, map/reduce, compaction.

On-device models have small context windows, so Kennel never assumes a whole
document fits. Long inputs go through ``chunk_text`` and ``map_reduce``;
conversation history is compacted with ``compact_history`` when the provider
reports a context overflow.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers.base import ProviderSession


# Token estimation constants. The on-device model gives us no token counter, so
# usage is estimated from text. Measured against Apple's tokenizer on English and
# Japanese samples: latin script runs about four characters per token, CJK about
# two. Tool output is only kept as a UTF-8 byte count, so it is divided by four.
CHARS_PER_TOKEN = 4.0
CJK_CHARS_PER_TOKEN = 2.0
BYTES_PER_TOKEN = 4.0

_CJK = re.compile(
    r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)


def estimate_tokens(text: str) -> float:
    """Estimate how many tokens ``text`` costs, counting CJK characters as denser.

    Deliberately cheap and approximate: it exists so a 4k window can be shown as
    a percentage, not to predict the tokenizer exactly.
    """
    if not text:
        return 0.0
    cjk = len(_CJK.findall(text))
    return cjk / CJK_CHARS_PER_TOKEN + (len(text) - cjk) / CHARS_PER_TOKEN


def estimate_tokens_from_bytes(size: int) -> float:
    """Estimate tokens for text we only kept the UTF-8 byte length of."""
    return max(0, size) / BYTES_PER_TOKEN


_NARRATION_PHRASES = re.compile(
    # announcing steps instead of taking them
    r"(実行します|検索します|探します|読み込みます|確認します|行います|させてください|してみます|しましょう|以下のコマンド|次のコマンド|"
    r"確認する必要があります|探す必要があります|まとめる必要があります|"
    r"\bI(?:'ll| will) (?:run|search|use|call|look|check|read|execute|find)\b|"
    r"\bLet me (?:run|search|check|read|look|find)\b|\bI am going to (?:run|search|use|call)\b|"
    r"\bI (?:would|can) (?:run|search|use|check|look)\b|"
    # asking which file to look at, or for permission to look, instead of looking
    r"どのファイル|確認できますか|確認すればよい|確認してもよろしい|探してもよろしい|探してみましょうか|確認しましょうか|"
    r"\bwhich files? (?:should|would|do)\b|\bshould I (?:search|look|check|read|find|open)\b|"
    r"\b(?:would|do) you (?:like|want) me to (?:search|look|check|read|find|open)\b|"
    r"\bcan I (?:search|check|look|read)\b)",
    re.IGNORECASE,
)


def looks_like_tool_narration(text: str, tool_names: Iterable[str]) -> str | None:
    """The rule name if an answer describes tool steps instead of taking them, else ``None``.

    Used only for turns in which no tool was actually called. Returns which rule
    fired so callers (``Session._should_nudge``) can report it on ``model.nudged``:

    - ``"code_block"`` — a fenced code block (the model showed a command instead
      of running it)
    - ``"file_mention"`` — talking about files without having looked at any
    - ``"tool_name"`` — an explicit mention of an available tool by name
      (``"the read tool"``, ``"call glob"``), not a bare appearance of a common
      English word that happens to share a tool's name
    - ``"phrase"`` — an "I will run ..." announcement or a "which file should I
      check?" question, in English or Japanese

    An agent with file tools should look rather than ask. The return value is
    truthy exactly when the previous ``bool`` contract would have been ``True``,
    so existing ``assert looks_like_tool_narration(...)`` callers keep working.
    """
    if not text.strip():
        return None
    if "```" in text:
        return "code_block"
    # Talking about files without having looked at any is narration or a clarifying
    # question an agent with file tools should answer by looking.
    if "ファイル" in text or re.search(r"\bfiles?\b", text, re.IGNORECASE):
        return "file_mention"
    lowered = text.lower()
    # Require the tool name to appear next to "tool" (e.g. "the read tool", "call
    # the glob tool"): tool names like read/write/edit/shell are ordinary English
    # words, so a bare `\bname\b` match fires on unrelated sentences ("Please edit
    # the summary").
    if any(re.search(rf"\b(?:{re.escape(name)}\s+tool|tool\s+{re.escape(name)})\b", lowered) for name in tool_names):
        return "tool_name"
    return "phrase" if _NARRATION_PHRASES.search(text) else None


def truncate_text(text: str, max_bytes: int, marker: str = "\n... [output truncated]") -> tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text, False
    cut = data[: max(0, max_bytes - len(marker.encode()))].decode("utf-8", errors="ignore")
    return cut + marker, True


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 200) -> list[str]:
    """Split text into chunks of at most ``max_chars`` on paragraph/line boundaries.

    Consecutive chunks overlap by roughly ``overlap`` characters so sentences cut
    at a boundary are still seen whole in one of them.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = max(0, min(overlap, max_chars // 2))
    if len(text) <= max_chars:
        return [text] if text else []
    units: list[str] = []
    for para in text.split("\n\n"):
        if len(para) + 2 <= max_chars:
            units.append(para + "\n\n")
        else:
            for line in para.splitlines(keepends=True):
                while len(line) > max_chars:
                    units.append(line[:max_chars])
                    line = line[max_chars:]
                units.append(line)
            units.append("\n")
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(current) + len(unit) > max_chars and current:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        current += unit
    if current.strip():
        chunks.append(current)
    return [c.strip("\n") for c in chunks if c.strip()]


@dataclass
class HistoryTurn:
    prompt: str
    response: str


def compact_history(turns: Sequence[HistoryTurn], max_chars: int = 2000, per_item: int = 300) -> str:
    """Compress prior turns into a short text block for a fresh model session."""
    if not turns:
        return ""
    lines = ["Summary of the conversation so far (older context was compacted):"]
    for turn in turns:
        lines.append(f"- User asked: {_squash(turn.prompt, per_item)}")
        if turn.response:
            lines.append(f"  Assistant answered: {_squash(turn.response, per_item)}")
    text = "\n".join(lines)
    suffix = "\n... (truncated)"
    if len(text) > max_chars:
        text = text[: max_chars - len(suffix)].rstrip() + suffix
    return text


def _squash(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


async def map_reduce(
    open_session: Callable[[], Awaitable[ProviderSession]],
    chunks: Sequence[str],
    *,
    map_prompt: Callable[[int, int, str], str],
    reduce_prompt: Callable[[Sequence[str]], str],
) -> str:
    """Summarize each chunk in its own fresh session, then reduce the partials.

    ``open_session`` must return a *new* provider session each time so that no
    chunk inherits another chunk's context.
    """
    partials: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        session = await open_session()
        try:
            partials.append(await session.respond(map_prompt(index, len(chunks), chunk)))
        finally:
            await session.close()
    if len(partials) == 1:
        return partials[0]
    session = await open_session()
    try:
        return await session.respond(reduce_prompt(partials))
    finally:
        await session.close()
