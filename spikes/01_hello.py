"""Spike 01: import, availability, basic respond, streaming."""
import asyncio, time
import apple_fm_sdk as fm

async def main():
    model = fm.SystemLanguageModel()
    ok, reason = model.is_available()
    print("available:", ok, reason)
    session = fm.LanguageModelSession(instructions="Answer briefly.", model=model)
    t = time.time()
    r = await session.respond("Say hello in one short sentence.")
    print(f"respond ({time.time()-t:.2f}s): {r!r}")
    t = time.time(); first = None; last = ""
    async for snap in session.stream_response("Count from 1 to 5, separated by commas."):
        if first is None: first = time.time() - t
        last = snap
    print(f"stream: ttft={first:.2f}s total={time.time()-t:.2f}s final={last!r}")
    d = await session.transcript.to_dict()
    print("transcript entries:", [(e["role"]) for e in d["transcript"]["entries"]])

asyncio.run(main())
