# Architecture notes

```text
Application (CLI / custom Python app)
        |
        v
+---------------------------------------------+
| Kennel SDK                                  |
|  Agent -> Session -> ToolRunner             |
|  Workspace  PermissionManager  EventBus     |
+------------------+--------------------------+
                   |                  \
                   v                   v
            ModelProvider          Tools (glob grep read
            AppleProvider           write edit shell web)
            MockProvider
                   |
                   v
       apple_fm_sdk -> Foundation Models
```

## Who drives the loop

Apple's SDK runs the tool-calling loop natively: `LanguageModelSession.respond()` invokes
registered tools until the model produces a final answer. Kennel does not reimplement that
protocol. Instead, every Kennel tool is bridged into an `apple_fm_sdk.Tool` whose `call()`
forwards to `ToolRunner.invoke(name, arguments)`. The runner is the single place where
validation, guardrails (per-turn budget, repeated identical calls), permission checks,
execution, output bounding and events happen.

## Threading (verified against apple_fm_sdk 0.2.0)

- The SDK invokes `Tool.call()` on a worker thread with its own event loop. `ToolRunner`
  is therefore thread-safe and never awaits objects bound to the caller's loop.
- `stream_response()` blocks the loop it runs on between snapshots. `AppleProvider` runs
  every request on a dedicated model thread and forwards results (or snapshots) back to
  the caller's loop, which keeps the CLI responsive and makes `Ctrl-C` a clean task
  cancellation.
- Bridged SDK tool objects must stay referenced for the session lifetime; `AppleSession`
  holds them.

## Context

The provider maps the SDK's context-window error to `ContextLimitError`. `Session.run`
never retries the same request blindly: it compacts the conversation into a short summary,
opens a fresh provider session seeded with it, retries once, and otherwise fails with a
user-facing message. Long documents are handled by `kennel.context.chunk_text` and
`map_reduce` (see `examples/meeting_summary.py`).

## Events

`EventBus` delivers `session.*`, `model.*`, `tool.*`, `permission.*` and
`context.compacted` events synchronously to subscribers, possibly from a worker thread.
Events carry summaries, sizes and timings, never file contents. The CLI renderer is one
subscriber; `--trace` prints the raw events as JSON lines.
