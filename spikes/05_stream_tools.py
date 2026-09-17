"""Spike 05: does stream_response() work with tools? When do snapshots arrive relative to tool calls?"""
import asyncio, time, threading
from typing import Optional
import apple_fm_sdk as fm
from apple_fm_sdk.generation_schema import GenerationSchema
from apple_fm_sdk.generation_property import Property

FILES = {"transcripts/2026-09-16.txt": "Meeting 2026-09-16. Decided: ship v0.1 Friday. TODO: write release notes."}
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.2f}s]", *a, flush=True)

class ReadTool(fm.Tool):
    name = "read"; description = "Read a text file from the workspace by path."
    @property
    def arguments_schema(self):
        return GenerationSchema(type_class=type("ReadArgs", (), {}), description="read args", properties=[Property("path", str, "workspace-relative path")])
    async def call(self, args):
        log("  [read]", args.to_json(), "thread", threading.current_thread().name)
        await asyncio.sleep(1.0)  # simulate slow tool; does main loop stay responsive?
        return FILES.get(args.value(str, for_property="path"), "No such file")

TOOLS = [ReadTool()]

async def ticker():
    for _ in range(40):
        await asyncio.sleep(0.25); log("  tick (main loop alive)")

async def main():
    s = fm.LanguageModelSession(instructions="Use tools to inspect files instead of guessing.", tools=TOOLS)
    t = asyncio.create_task(ticker())
    n = 0
    async for snap in s.stream_response("Read transcripts/2026-09-16.txt and tell me the decision."):
        n += 1
        if n <= 3 or n % 10 == 0: log("snapshot", n, repr(snap[:60]))
    log("final snapshot count", n, "| last:", repr(snap))
    t.cancel()
    d = await s.transcript.to_dict(); log("roles:", [e["role"] for e in d["transcript"]["entries"]])

asyncio.run(main())
