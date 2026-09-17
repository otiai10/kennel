"""Reference workflow: structured meeting summary from a transcript file.

Kennel core does not know what a "meeting" is; this example shows how an
application layers a domain schema on top of the SDK: chunking for long
transcripts, ``Agent.run(prompt, schema=...)`` per chunk, and a structured reduce.

The agent is built with ``tools=[]``: the transcript text is passed in the prompt, so
there is nothing to look up and each chunk costs exactly one model request. Note that
``Agent`` prepends its default instructions (a glob/read procedure) to the domain
instructions below; replacing rather than appending them is a separate feature.

Usage::

    python examples/meeting_summary.py ~/meetings/transcripts/2026-09-16.txt
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kennel import Agent
from kennel.context import chunk_text
from kennel.providers.base import ModelProvider

NULL_STRINGS = {"", "null", "none", "undefined", "n/a", "unknown", "tbd", "not specified"}


def clean_optional(value: Any) -> str | None:
    """Guided generation sometimes returns 'undefined' for missing optionals."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in NULL_STRINGS else text


@dataclass
class ActionItem:
    task: str
    owner: str | None = None
    due_date: str | None = None


@dataclass
class MeetingSummary:
    title: str | None
    summary: str
    decisions: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    unresolved_topics: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeetingSummary:
        items = []
        for raw in data.get("action_items") or []:
            if isinstance(raw, dict) and clean_optional(raw.get("task")):
                items.append(ActionItem(task=str(raw["task"]).strip(), owner=clean_optional(raw.get("owner")), due_date=clean_optional(raw.get("due_date"))))
        return cls(
            title=clean_optional(data.get("title")),
            summary=str(data.get("summary") or "").strip(),
            decisions=[str(d).strip() for d in data.get("decisions") or [] if clean_optional(d)],
            action_items=items,
            unresolved_topics=[str(t).strip() for t in data.get("unresolved_topics") or [] if clean_optional(t)],
        )


MEETING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Structured summary of a meeting transcript",
    "properties": {
        "title": {"type": "string", "description": "Meeting title or topic, or 'unknown' if not stated"},
        "summary": {"type": "string", "description": "Summary in at most three sentences"},
        "decisions": {"type": "array", "items": {"type": "string"}, "description": "Decisions explicitly made"},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What has to be done"},
                    "owner": {"type": "string", "description": "Person explicitly assigned, or 'unknown'"},
                    "due_date": {"type": "string", "description": "Due date explicitly stated, or 'unknown'"},
                },
                "required": ["task", "owner", "due_date"],
            },
        },
        "unresolved_topics": {"type": "array", "items": {"type": "string"}, "description": "Questions left open"},
    },
    "required": ["title", "summary", "decisions", "action_items", "unresolved_topics"],
}

INSTRUCTIONS = (
    "You extract facts from meeting transcripts. Only report decisions, tasks, owners and dates that are "
    "explicitly stated in the text. If an owner or date is not stated, use 'unknown'. Never invent details."
)


def build_agent(provider: ModelProvider | None = None, workspace: str | Path = ".") -> Agent:
    """An extraction-only agent: no tools, the domain instructions, one request per call."""
    return Agent(workspace, tools=[], instructions=INSTRUCTIONS, provider=provider)


async def extract(agent: Agent, prompt: str) -> MeetingSummary:
    result = await agent.run(prompt, schema=MEETING_SCHEMA)
    return MeetingSummary.from_dict(result.structured_output or {})


async def summarize_transcript(provider: ModelProvider | None, text: str, *, chunk_chars: int = 6000) -> MeetingSummary:
    """Map each chunk to a structured partial, then reduce into one MeetingSummary."""
    agent = build_agent(provider)
    chunks = chunk_text(text, max_chars=chunk_chars, overlap=200)
    partials: list[MeetingSummary] = []
    for index, chunk in enumerate(chunks, 1):
        prompt = f"Transcript part {index} of {len(chunks)}:\n\n{chunk}\n\nExtract the summary, decisions, action items and unresolved topics from this part."
        partials.append(await extract(agent, prompt))
    if len(partials) == 1:
        return partials[0]
    merged = MeetingSummary(
        title=next((p.title for p in partials if p.title), None),
        summary="\n".join(p.summary for p in partials if p.summary),
        decisions=[d for p in partials for d in p.decisions],
        action_items=[a for p in partials for a in p.action_items],
        unresolved_topics=[t for p in partials for t in p.unresolved_topics],
    )
    reduced = await extract(
        agent,
        "These are partial extractions from consecutive parts of one meeting. Merge them: remove duplicates, "
        "keep every distinct decision, task, owner and date exactly as given, and write one overall summary of "
        f"at most three sentences.\n\n{json.dumps(asdict(merged), ensure_ascii=False, indent=1)}",
    )
    if not reduced.decisions and merged.decisions:
        reduced.decisions = merged.decisions
    if not reduced.action_items and merged.action_items:
        reduced.action_items = merged.action_items
    return reduced


async def main(path: str) -> None:
    agent = build_agent(workspace=Path(path).parent)
    agent.check_availability()
    text = Path(path).read_text(encoding="utf-8")
    summary = await summarize_transcript(agent.provider, text)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/meeting_summary.py <transcript.txt>")
    asyncio.run(main(sys.argv[1]))
