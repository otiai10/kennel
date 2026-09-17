"""Spike 02: guided generation with @generable and with a programmatic GenerationSchema."""
import asyncio
from typing import List, Optional
import apple_fm_sdk as fm
from apple_fm_sdk.generation_schema import GenerationSchema
from apple_fm_sdk.generation_property import Property

@fm.generable("An action item extracted from a meeting")
class ActionItem:
    task: str = fm.guide("The task to do")
    owner: Optional[str] = fm.guide("Owner name if explicitly stated, otherwise null")

@fm.generable("A meeting summary")
class MeetingSummary:
    summary: str = fm.guide("Two sentence summary")
    decisions: List[str] = fm.guide("Decisions made")
    action_items: List[ActionItem] = fm.guide("Action items")

TRANSCRIPT = """Alice: We decided to ship v0.1 on Friday.
Bob: OK. I'll write the release notes.
Alice: And someone needs to update the README, we haven't decided who."""

async def main():
    session = fm.LanguageModelSession(instructions="Extract facts. Never invent owners.")
    ms = await session.respond(f"Summarize this meeting:\n{TRANSCRIPT}", generating=MeetingSummary)
    print("type:", type(ms).__name__)
    print("summary:", ms.summary)
    print("decisions:", ms.decisions)
    print("action_items:", [(a.task, a.owner) for a in ms.action_items])

    # Programmatic schema (no decorator) - what a generic tool adapter would need
    class _Args: pass
    schema = GenerationSchema(type_class=_Args, description="Args", properties=[
        Property(name="pattern", type_class=str, description="glob pattern"),
        Property(name="max_results", type_class=Optional[int], description="max results"),
    ])
    print("schema dict:", schema.to_dict())
    s2 = fm.LanguageModelSession()
    gc = await s2.respond("Produce args to find all markdown files, limit 10", schema=schema)
    print("generated:", gc.to_json(), "| pattern=", gc.value(str, for_property="pattern"), "| max=", gc.value(for_property="max_results"))

asyncio.run(main())
