---
kind: guide
---

# Architecture notes

The principles behind these choices are in [constitution.md](constitution.md). This page is the map of where each one is enforced; when it disagrees with the code, the code is right.

```text
Application (CLI / custom Python app)
        |
        v
+--------------------------------------------------------------+
| Kennel SDK                                                   |
|  Agent -> Session -> ToolRunner                              |
|  Workspace  PermissionManager (rules)  Hooks  EventBus       |
+---------------------+----------------------------------------+
                      |                    \
                      v                     v
               ModelProvider           Tools (glob grep read
               AppleProvider            write edit shell web)
               MockProvider
                      |
                      v
          apple_fm_sdk -> Foundation Models
```

## Who drives the loop

Apple's SDK runs the tool-calling loop natively: `LanguageModelSession.respond()` invokes registered tools until the model produces a final answer. Kennel does not reimplement that protocol. Every Kennel tool is bridged into an `apple_fm_sdk.Tool` whose `call()` forwards to `ToolRunner.invoke(name, arguments)`, and the runner is the single place where validation, guardrails (per-turn budget, repeated identical calls), hooks, permission checks, execution, output bounding and events happen (`src/kennel/runner.py`).

Guided generation goes through the same loop: `Session.run(schema=)` calls `ProviderSession.respond_structured()`, and the provider still invokes tools inside that one request (verified on device, `tests/integration/test_apple.py`).

## One call, in order

For every tool call `ToolRunner.invoke` does, in this order:

1. validate and coerce the arguments (`Tool.validate`)
2. guardrails: per-turn budget and repeated-call detection
3. `before_tool` hooks, which may `Deny` (recorded as `blocked`, `tool.blocked` emitted) or `Allow(updated_arguments=)` (re-validated)
4. the permission decision on the arguments the tool will actually run with: `PermissionManager.decision_for(name, kind, arguments, tool.match_rule)`, then `decide()` prompts if it says `ask`
5. `tool.execute`
6. `after_tool` hooks, which may replace the result; the replacement is bounded like any other
7. output bounding and the `tool.completed` / `tool.failed` event

Step 4 is the only place the permission precedence lives (`src/kennel/permissions.py`, `decision_for`): `deny` always wins, then a matching specifier rule beats the bare tool rule, then `ask` beats `allow`. A permission mode supplies the defaults the rules layer on; what a specifier means is decided by the tool (`Tool.match_rule`, glob syntax in `src/kennel/rules.py`). Pinned by `tests/unit/test_permission_rules.py` and `tests/unit/test_hooks.py`.

## Threading (verified against apple_fm_sdk 0.2.0)

- The SDK invokes `Tool.call()` on a worker thread with its own event loop. `ToolRunner`, hooks and the permission manager are therefore thread-safe and never await objects bound to the caller's loop.
- `stream_response()` blocks the loop it runs on between snapshots. `AppleProvider` runs every request on a dedicated model thread and forwards results (or snapshots) back to the caller's loop.
- Bridged SDK tool objects must stay referenced for the session lifetime; `AppleSession` holds them.
- `Session.stream()` relays bus events onto the consumer's loop through an `asyncio.Queue`, so an application never sees a worker thread. `Session.interrupt()` posts the cancellation to the loop that started the turn and is safe from any thread; the CLI's Ctrl-C uses the same call.

## Context

The provider maps the SDK's context-window error to `ContextLimitError`. `Session.run` never retries the same request blindly: it compacts the conversation into a short summary, opens a fresh provider session seeded with it, retries once, and otherwise fails with a user-facing message. `Session.compact()` exposes the same step on demand.

`Session.context_usage()` reports how full the window is. `ProviderInfo.context_window_tokens` declares the window (4096 on the on-device model); `ProviderSession.usage()` may report a real count, otherwise Kennel estimates from the text in the live provider session and says so with `ContextUsage.estimated`. `AgentResult.usage` is filled only from a real count, never from the estimate. Long documents are handled by `kennel.context.chunk_text` and `map_reduce` (see `examples/meeting_summary.py`).

## Events and hooks

`EventBus` delivers `session.*`, `model.*`, `tool.*`, `permission.*` and `context.compacted` events synchronously to subscribers, possibly from a worker thread. The list of types is `EventType` in `src/kennel/events.py`. Events carry summaries, sizes and timings, never file contents; a subscriber that raises is logged and ignored. Consumers: the CLI renderer, `--trace` (raw events on stderr), `--output-format stream-json` (documented schema in [output-format.md](output-format.md)) and `Session.stream()`.

Hooks (`src/kennel/hooks.py`) are the opposite kind of callback: they decide. `before_tool` / `after_tool` run inside `ToolRunner.invoke` and a hook that raises fails the call; `before_prompt` runs in `Session.run` and appends context to the user's prompt (not to Kennel's own re-prompts). They execute in-process with the application's own trust level, see [SECURITY.md](../SECURITY.md).

## Providers

`ModelProvider.create_session()` returns a `ProviderSession` with `respond`, `stream`, optional `respond_structured` and `usage`, and `close`. `ProviderInfo` declares `name`, `model`, `mode` (`local` means inference and data stay on this machine; anything else must say so) and `context_window_tokens`. `MockProvider` scripts turns, tool calls, structured values and reported usage for the test suite (`tests/providers/`).
