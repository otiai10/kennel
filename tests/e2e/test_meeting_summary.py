"""The reference transcript workflow, driven by MockProvider structured responses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from meeting_summary import MeetingSummary, summarize_transcript  # noqa: E402

from kennel import MockProvider  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "meeting_project" / "transcripts" / "2026-09-16.txt"


def test_from_dict_normalizes_missing_owners():
    s = MeetingSummary.from_dict(
        {
            "title": "undefined",
            "summary": " Ship it. ",
            "decisions": ["Ship v0.1 on Friday", "null"],
            "action_items": [{"task": "Write release notes", "owner": "undefined", "due_date": "unknown"}, {"task": "", "owner": "x"}],
            "unresolved_topics": ["web tool"],
        }
    )
    assert s.title is None and s.summary == "Ship it." and s.decisions == ["Ship v0.1 on Friday"]
    assert len(s.action_items) == 1 and s.action_items[0].owner is None and s.action_items[0].due_date is None


async def test_single_chunk_uses_one_structured_call():
    structured = [{"title": "Release planning", "summary": "S", "decisions": ["Ship Friday"], "action_items": [{"task": "README security section", "owner": "Bob", "due_date": "Friday"}], "unresolved_topics": ["release notes owner"]}]
    provider = MockProvider(structured=structured)
    summary = await summarize_transcript(provider, FIXTURE.read_text())
    assert summary.decisions == ["Ship Friday"] and summary.action_items[0].owner == "Bob"
    assert len(provider.sessions) == 1 and FIXTURE.read_text()[:40] in provider.sessions[0].prompts[0]


async def test_long_transcript_is_chunked_and_reduced():
    text = "\n\n".join(f"Speaker {i}: we talked about topic {i} at length. " * 5 for i in range(60))
    structured = [
        {"title": "unknown", "summary": "part one", "decisions": ["D1"], "action_items": [], "unresolved_topics": []},
        {"title": "Big meeting", "summary": "part two", "decisions": ["D2"], "action_items": [{"task": "T", "owner": "unknown", "due_date": "unknown"}], "unresolved_topics": ["U"]},
        {"title": "Big meeting", "summary": "merged", "decisions": ["D1", "D2"], "action_items": [{"task": "T", "owner": "unknown", "due_date": "unknown"}], "unresolved_topics": ["U"]},
    ]
    provider = MockProvider(structured=structured)
    summary = await summarize_transcript(provider, text, chunk_chars=8000)
    assert len(provider.sessions) == 3  # two map calls + one reduce
    assert "Transcript part 1 of 2" in provider.sessions[0].prompts[0]
    assert summary.summary == "merged" and summary.decisions == ["D1", "D2"]
    assert summary.action_items[0].owner is None and summary.title == "Big meeting"
