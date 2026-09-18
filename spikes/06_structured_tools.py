"""Spike 06: does guided generation work on a session that has tools registered?

The question Kennel has to answer before promoting structured output to
``Session.run(schema=)``: can one request both call tools and return a schema-shaped
value, or does it take two turns (tools first, then guided generation over the same
session)?

Three probes, each printing what actually happened:

1. one request with ``json_schema=`` on a session that has a tool the prompt needs
2. the same, where the answer is already in the instructions (no tool needed)
3. two turns on one session: a normal tool-using turn, then ``json_schema=``

Run on an Apple Silicon Mac with Apple Intelligence enabled::

    uv run python spikes/06_structured_tools.py
"""

import asyncio

import apple_fm_sdk as fm
from apple_fm_sdk.generation_property import Property
from apple_fm_sdk.generation_schema import GenerationSchema

TRANSCRIPT = """Alice: We decided to ship v0.1 on Friday.
Bob: OK. I'll write the release notes.
Alice: The README owner is still undecided."""

SUMMARY_SCHEMA = {
    "type": "object",
    "title": "MeetingSummary",
    "x-order": ["summary", "decisions"],
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string", "description": "One sentence"},
        "decisions": {"type": "array", "items": {"type": "string"}, "description": "Decisions made"},
    },
    "required": ["summary", "decisions"],
}

calls: list[str] = []


def make_tool():
    """A `read_transcript` tool; the SDK calls it on its own thread + event loop."""

    class _Args:
        pass

    schema = GenerationSchema(
        type_class=_Args,
        description="Arguments for read_transcript",
        properties=[Property("path", str, "Path of the transcript to read")],
    )

    class ReadTranscript(fm.Tool):
        name = "read_transcript"
        description = "Read the text of a meeting transcript by path."

        @property
        def arguments_schema(self):
            return schema

        async def call(self, args) -> str:
            raw = args.value()
            calls.append(str(dict(raw) if isinstance(raw, dict) else raw))
            return TRANSCRIPT

    return ReadTranscript()


async def probe(label: str, body) -> None:
    calls.clear()
    print(f"\n--- {label}")
    try:
        result = await body()
    except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
        print(f"  RAISED {type(exc).__name__}: {exc}")
        print(f"  tool calls: {calls}")
        return
    print(f"  value: {result}")
    print(f"  tool calls: {calls}")


async def main() -> None:
    model = fm.SystemLanguageModel()
    ok, reason = model.is_available()
    if not ok:
        raise SystemExit(f"model unavailable: {reason}")

    # 1. one request that needs a tool AND a schema
    async def one_shot_needs_tool():
        tool = make_tool()  # keep referenced: the SDK holds a raw pointer
        session = fm.LanguageModelSession(instructions="Extract facts from transcripts.", model=model, tools=[tool])
        content = await session.respond(
            "Read the transcript at meeting.txt and extract the summary and the decisions.",
            json_schema=SUMMARY_SCHEMA,
        )
        return content.value()

    await probe("1. tools + json_schema in one request (tool needed)", one_shot_needs_tool)

    # 2. schema on a tool-registered session where no tool is needed
    async def one_shot_no_tool_needed():
        tool = make_tool()
        session = fm.LanguageModelSession(instructions=f"Extract facts. Transcript:\n{TRANSCRIPT}", model=model, tools=[tool])
        content = await session.respond("Extract the summary and the decisions.", json_schema=SUMMARY_SCHEMA)
        return content.value()

    await probe("2. tools registered but not needed + json_schema", one_shot_no_tool_needed)

    # 3. two turns on one session: tools first, then guided generation
    async def two_phase():
        tool = make_tool()
        session = fm.LanguageModelSession(instructions="Extract facts from transcripts.", model=model, tools=[tool])
        first = await session.respond("Read the transcript at meeting.txt and tell me what was decided.")
        print(f"  phase 1 text: {str(first)[:120]!r}")  # respond() without a schema returns a str
        content = await session.respond(
            "Now give that answer again as data following the schema. Use only what you just read.",
            json_schema=SUMMARY_SCHEMA,
        )
        return content.value()

    await probe("3. two turns on one session (tools, then json_schema)", two_phase)


if __name__ == "__main__":
    asyncio.run(main())
