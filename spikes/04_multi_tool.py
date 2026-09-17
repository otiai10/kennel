"""Spike 04: multiple tools, multi-step tool calling, cancellation, and context-size error."""
import asyncio
from typing import Optional
import apple_fm_sdk as fm
from apple_fm_sdk.generation_schema import GenerationSchema
from apple_fm_sdk.generation_property import Property

FILES = {
    "transcripts/2026-09-15.txt": "Meeting 2026-09-15. Decided: adopt pytest. TODO: Bob writes CI config.",
    "transcripts/2026-09-16.txt": "Meeting 2026-09-16. Decided: ship v0.1 Friday. TODO: write release notes (owner undecided).",
}
def schema(name, desc, props):
    cls = type(name, (), {})
    return GenerationSchema(type_class=cls, description=desc, properties=props)

class GlobTool(fm.Tool):
    name = "glob"; description = "Find files by glob pattern in the workspace."
    @property
    def arguments_schema(self): return schema("GlobArgs", "glob args", [Property("pattern", str, "glob pattern")])
    async def call(self, args):
        print("  [glob]", args.to_json()); return "\n".join(FILES)

class ReadTool(fm.Tool):
    name = "read"; description = "Read a text file from the workspace by path."
    @property
    def arguments_schema(self): return schema("ReadArgs", "read args", [Property("path", str, "workspace-relative path"), Property("start_line", Optional[int], "1-based start line")])
    async def call(self, args):
        p = args.value(str, for_property="path"); print("  [read]", args.to_json())
        if p not in FILES: raise FileNotFoundError(f"No such file: {p}")
        return FILES[p]

TOOLS = [GlobTool(), ReadTool()]

async def main():
    session = fm.LanguageModelSession(
        instructions="You are Kennel, a local tool-using assistant. Inspect files with tools instead of guessing. Never claim to have read a file unless you read it.",
        tools=TOOLS)
    r = await session.respond("Find the latest meeting transcript, read it, and list the decisions and TODOs.")
    print("response:", r)
    d = await session.transcript.to_dict()
    print("roles:", [e["role"] for e in d["transcript"]["entries"]])
    # cancellation
    print("--- cancellation ---")
    s2 = fm.LanguageModelSession()
    task = asyncio.create_task(s2.respond("Write a 2000 word essay about the ocean."))
    await asyncio.sleep(0.5); task.cancel()
    try: await task
    except asyncio.CancelledError: print("cancelled OK; is_responding=", s2.is_responding)
    r3 = await s2.respond("Say 'ok'."); print("after cancel:", r3)
    # context overflow
    print("--- context overflow ---")
    s3 = fm.LanguageModelSession()
    try:
        await s3.respond("Summarize: " + ("lorem ipsum dolor sit amet " * 3000))
        print("no error?!")
    except fm.ExceededContextWindowSizeError as e:
        print("ExceededContextWindowSizeError:", str(e)[:120])
    except Exception as e:
        print("other error:", type(e).__name__, str(e)[:200])

asyncio.run(main())
