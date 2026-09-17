"""Spike 03: a Tool with a programmatic schema; inspect threading/loop context and error reporting."""
import asyncio, threading
from typing import Optional
import apple_fm_sdk as fm
from apple_fm_sdk.generation_schema import GenerationSchema
from apple_fm_sdk.generation_property import Property

MAIN_THREAD = threading.get_ident()
calls = []

class _ListFilesArgs: pass

class ListFilesTool(fm.Tool):
    name = "list_files"
    description = "Lists files in the workspace matching a glob pattern. Use this instead of guessing file names."
    @property
    def arguments_schema(self) -> GenerationSchema:
        return GenerationSchema(type_class=_ListFilesArgs, description="Arguments for list_files", properties=[
            Property(name="pattern", type_class=str, description="Glob pattern such as **/*.md"),
            Property(name="max_results", type_class=Optional[int], description="Maximum number of results"),
        ])
    async def call(self, args: fm.GeneratedContent) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        info = dict(thread=threading.get_ident(), main=threading.get_ident()==MAIN_THREAD, loop=id(loop) if loop else None, raw=args.to_json())
        calls.append(info)
        print("  [tool call]", info)
        pattern = args.value(str, for_property="pattern")
        if pattern == "FAIL":
            raise ValueError("simulated failure")
        return "transcripts/2026-09-15.txt\ntranscripts/2026-09-16.txt\nnotes/todo.md"

async def main():
    print("main loop:", id(asyncio.get_running_loop()))
    tool = ListFilesTool()
    session = fm.LanguageModelSession(
        instructions="You are a local assistant. Use tools to inspect files instead of guessing.",
        tools=[tool])
    r = await session.respond("Which transcript files exist in this workspace? List them.")
    print("response:", r)
    print("calls:", len(calls))
    d = await session.transcript.to_dict()
    for e in d["transcript"]["entries"]:
        print(" ", e["role"], "| toolCalls=", [tc.get("toolName") for tc in e.get("toolCalls", [])] if e.get("toolCalls") else "", "| toolName=", e.get("toolName",""), "| text=", [c.get("text","")[:80] for c in e.get("contents", [])])
    # Error path: ask for something that yields FAIL? Not controllable; instead call tool directly-ish via a second session prompt
    print("--- error path ---")
    s2 = fm.LanguageModelSession(instructions="Always call list_files with pattern exactly 'FAIL' first.", tools=[ListFilesTool()])
    try:
        r2 = await s2.respond("Call list_files with pattern FAIL and tell me what happened.")
        print("response2:", r2)
    except Exception as e:
        print("raised:", type(e).__name__, e)

asyncio.run(main())
