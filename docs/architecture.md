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
               AppleProvider            write edit shell web fetch)
               LlamaServerProvider
               LlamaCppProvider
               MockProvider
                      |
              +-------+--------+
              v                v
       apple_fm_sdk       ChatLoopSession -> ChatTransport
              |                                  |
              v                        +---------+---------+
     Foundation Models                 v                   v
                              llama-server (HTTP)   llama-cpp (in-process)
```

## Who drives the loop

There are two families of provider, and the difference is who runs the tool-calling loop.

**Native loop.** Apple's SDK runs it: `LanguageModelSession.respond()` invokes registered tools until the model produces a final answer, and Kennel does not reimplement that protocol. Every Kennel tool is bridged into an `apple_fm_sdk.Tool` whose `call()` forwards to `ToolRunner.invoke(name, arguments)`.

**Shared loop.** A chat-completions model does not run any loop, so Kennel runs one — once, in `ChatLoopSession` (`src/kennel/providers/chatloop.py`). A provider in this family implements only a `ChatTransport` (messages in, `ChatChunk`s out) and gets the loop, the message bookkeeping and the termination rules from there; `LlamaServerProvider` and `LlamaCppProvider` are both in it. Its rules:

- every request is streamed (`stream: true`), so `respond()`, `stream()` and `respond_structured()` are three wrappers around one loop and cannot disagree about the text;
- streamed tool-call fragments are merged by their `index`; a thinking model's `reasoning_content` is dropped by the transport and never reaches the answer;
- a tool call whose arguments are not a JSON object is answered with an error message rather than forwarded with a guessed `{}` — nothing reaches a tool except through `invoke`;
- when tool calls run out is *not* decided here. The loop reads `ToolRunner.no_more_tool_rounds` through the invoker it was handed (`_limit_reached`, which looks at the bound method's `__self__`, because `create_session` passes a bare callable); that one flag covers both ways out — the per-turn budget (`limit_hit`) and a model wedged repeating one call (`repeat_wedged`) — and once they have run out one further request is allowed, and a further tool request ends the turn with the text so far. When the loop returns from that last round it tells the runner (`_report_rounds_stopped` → `ToolRunner.stop_tool_rounds`), and `Session.run` reports the turn as `stop_reason` `tool_limit` whichever way the calls ran out — the one place a consumer looks to learn that the runtime cut the turn short (#66); the guardrail's `blocked` records say which. `repeat_wedged` alone is not that fact: on Apple, whose SDK runs its own loop, a narrowed call still runs after the wedge and the model may go on to answer, so such a turn stays `end_turn`. `tool_choice: "none"` is deliberately not used to force an end: llama.cpp then leaks raw `<tool_call>` text into the answer.

The two members of that family differ in *who renders the prompt*. A server does it (llama-server with `--jinja`), and hands back parsed `tool_calls` plus a separate `reasoning_content` the transport drops. In-process, llama-cpp-python does not: its generic handler passes `tools` to the chat template but never parses the answer, so `LlamaCppProvider` renders the GGUF's own `tokenizer.chat_template` with `Jinja2ChatFormatter` and parses the raw `<tool_call>{json}</tool_call>` itself, incrementally, because those tags arrive split across chunks (`_CompletionParser`). That parser lives with the provider, not in `chatloop.py`: it is the only transport that ever sees markup. The consequence is a supported wire format — the Qwen3/Hermes one — and a place to set a default: `chat_template_kwargs` is `{"enable_thinking": False}`, because Qwen3 otherwise thinks for a few hundred tokens before every tool call. Rendering has one requirement an HTTP body does not: a chat template reads `message.content` directly, so every message is rendered with that key present even when the assistant only asked for tools.

Either way `ToolRunner` is the single place where validation, guardrails (per-turn budget, repeated identical calls), hooks, permission checks, execution, output bounding and events happen (`src/kennel/runner.py`).

Guided generation goes through the same loop: `Session.run(schema=)` calls `ProviderSession.respond_structured()`, and the provider still invokes tools inside that one request (verified on device, `tests/integration/test_apple.py`).

## One call, in order

For every tool call `ToolRunner.invoke` does, in this order:

1. validate and coerce the arguments (`Tool.validate`)
2. guardrails: per-turn budget and repeated-call detection. A call that repeats one already made among the last few is refused instead of run (`max_repeats`); `max_repeat_refusals` such refusals in a row (reset by any call that is not one) make the runner give up on that call for the rest of the turn — `repeat_wedged`, sticky for the turn, and the call's answer becomes the same terminal instruction a spent budget gives. A call with different arguments is still run: narrowing the request is what the notice asked for. Without it a model that ignores a refusal spends the whole budget on one call and the user loses the turn (#58); with it a chat-style loop ends the turn, and a provider running its own loop (Apple) at least receives that instruction early, since a tool callback cannot stop its loop
3. `before_tool` hooks, which may `Deny` (recorded as `blocked`, `tool.blocked` emitted) or `Allow(updated_arguments=)` (re-validated here, re-matched in step 4)
4. the permission decision on the arguments the tool will actually run with: `PermissionManager.decision_for(name, kind, arguments, tool.match_rule)`, then `decide()` prompts if it says `ask`. A prompter that answers `Allow(updated_arguments=)` has its arguments re-validated and passed through `decision_for` again; only `deny` counts there (status `denied`, `permission.denied`), because the prompter approved exactly those arguments and is not asked twice (#73). "Allow for this session" — a hook's or the prompter's — comes back as `PermissionOutcome.remember_session` and is granted here, after every check has passed, with `pm.grant_session(name, tool.session_scope(args))` on the final arguments; `decide(remember=False, session_approved=...)` is how the runner keeps that grant (and a hook's session approval) inside `PermissionManager` without recording it early
5. `tool.execute`
6. `after_tool` hooks, which may replace the result; the replacement is bounded like any other
7. output bounding and the `tool.completed` / `tool.failed` event. `tool.completed` carries the result's size in bytes and in estimated tokens, plus `window_exceeded` for the case where that one result is already bigger than the provider's declared window — the runner is where the bytes are counted, so it is where the comparison belongs, and the CLI only draws it
8. the window guard: a result that `window_exceeded` marks is withheld and the model receives a one-line "this did not fit, ask for a smaller part" instruction in its place (`withheld` on the record and the event). Handing such a result over does not shorten the conversation, it fails the request — Apple answers status 255 — so the turn survives instead. The measured sizes still describe the result the tool produced, `tools.max_output_bytes` is not changed by the window (an explicit value stays the only output limit, and lowering it below the window brings truncated content back), and `Session.context_usage()` skips what was withheld

Step 4 is the only place the permission precedence lives (`src/kennel/permissions.py`, `decision_for`): `deny` always wins, then a matching specifier rule beats the bare tool rule, then `ask` beats `allow`. A permission mode supplies the defaults the rules layer on; what a specifier means is decided by the tool (`Tool.match_rule`, glob syntax in `src/kennel/rules.py`). Pinned by `tests/unit/test_permission_rules.py` and `tests/unit/test_hooks.py`.

## Threading (Apple verified against apple_fm_sdk 0.2.0, llama-server against 0.4.1, llama-cpp against llama-cpp-python 0.3.35)

- The SDK invokes `Tool.call()` on a worker thread with its own event loop. `ToolRunner`, hooks and the permission manager are therefore thread-safe and never await objects bound to the caller's loop.
- `stream_response()` blocks the loop it runs on between snapshots. `AppleProvider` runs every request on a dedicated model thread and forwards results (or snapshots) back to the caller's loop.
- Bridged SDK tool objects must stay referenced for the session lifetime; `AppleSession` holds them.
- `LanguageModelSession.respond()` returns a `str` without a schema and, with one, a content object whose `.value()` holds the structured result; `AppleSession` branches on that and callers must not assume one type.
- `Session.stream()` relays bus events onto the consumer's loop through an `asyncio.Queue`, so an application never sees a worker thread. `Session.interrupt()` posts the cancellation to the loop that started the turn and is safe from any thread; the CLI's Ctrl-C uses the same call.
- `LlamaCppProvider` is the same shape again: `create_completion` is a blocking generator, so it runs on a worker thread (`kennel-llama-cpp`) that feeds a queue. Cancelling sets an event the worker checks *between tokens*, so `interrupt()` lands at the next token boundary and not during prompt evaluation (10.8 s for a 6,752-token prompt). One `Llama` is one KV cache, so the provider holds a `threading.Lock` that the worker takes for the whole generation: the next `run()` starts only once the previous one has really stopped. The model is loaded on the first `create_session()` through `asyncio.to_thread` (so the caller's loop stays free) and shared by every session after that; it belongs to the provider, so closing a session does not unload it.
- `LlamaServerProvider` has the same shape for a different reason: `http.client` is blocking, so each request reads the stream on a worker thread and hands parsed events to the caller's loop through a queue. Cancelling the turn *shuts the socket down* rather than closing the response — the reading thread holds the buffered reader's lock, so closing it from the event loop would block the loop for as long as the request would have taken. Each request owns its connection, so an aborted one is never reused.

## Context

The provider maps the SDK's context-window error to `ContextLimitError`. `Session.run` never retries the same request blindly: it compacts the conversation into a short summary, opens a fresh provider session seeded with it, retries once, and otherwise fails with a user-facing message. `Session.compact()` exposes the same step on demand. A turn that fails for any other reason is recorded too: it lands in the history with `stop_reason "error"` and the tool calls it made, its live provider session is retired through the same compaction once the turn has started a generation call (the provider does not say whether the failed prompt stayed in its transcript, so the estimate must not keep counting it; a failure before that — a raising `before_prompt` hook — sent nothing and keeps the session, #72), and the facts of the turn — provider status, tool output bytes, the window before the request — go out on `session.failed` and stay on `Session.last_failure`. The estimated-window bookkeeping follows the provider session rather than the history: with no live session, `context_usage()` reports what the *next* one will be opened with (the instructions plus a summary of the whole history), because `Session._provider` is only ever called with that summary. Deriving it from "is a session live?" in one place, instead of resetting it in each of the five callers of `Session._drop_provider`, is what makes `compact()`, a failed turn, `clear()`, `Session.interrupt()` and a turn timeout agree with no exceptions (#45, #52) — otherwise a reader of `context_usage()` between the drop and the next request — the failed turn's own report, `/status` right after Ctrl-C — would still see the pre-drop size.

A single tool result can be bigger than the whole window, which is why `ToolRunner.invoke` withholds it (step 8 above) rather than letting the provider reject the request. `Session.context_usage()` reports how full the window is. `ProviderInfo.context_window_tokens` declares the window (4096 on the on-device model; whatever `/props` reports for a llama-server); `ProviderSession.usage()` may report a real count, otherwise Kennel estimates from the text in the live provider session and says so with `ContextUsage.estimated`, and why with `ContextUsage.estimate_reason` (`"not_reported"`: a session is live but the provider counts nothing; `"no_live_session"`: no session is live, so the figure is what the next one opens with). `AgentResult.usage` is filled only from a real count, never from the estimate. llama-server is the first provider that really counts: its `stream_options.include_usage` chunk is what `usage()` returns, so `estimated` is False there and stays True on Apple. `llama-cpp` counts too, with its own tokenizer: the prompt count is the length of the token list it was handed, and the output count is that tokenizer re-reading the generated text, because a streamed completion reports no counts and its chunks are not one token each.

The estimate's coefficients (`kennel.context`: `CHARS_PER_TOKEN`, `CJK_CHARS_PER_TOKEN`, `BYTES_PER_TOKEN`) cannot be read off a tokenizer, because the on-device model exposes none. They are calibrated instead by finding the size at which a `read` result stops fitting the declared 4,096-token window, which is measurable from the outside. Measured on macOS 27 with Apple Intelligence on (2026-09-22, #53), on a Japanese minutes-style file and a latin control, with `instructions` costing 331 estimated tokens:

| read result | characters (CJK) | on the device | bytes/4 | chars, CJK 2.0 | chars, CJK 1.5 | chars, CJK 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 14,205 B Japanese | 5,723 (4,241) | answered | 3,551 | 2,491 | 3,198 | 4,612 |
| 15,642 B Japanese | 6,306 (4,668) | answered | 3,910 | 2,744 | 3,522 | 5,078 |
| 16,571 B Japanese | 6,685 (4,943) | withheld by the guard | 4,143 | 2,907 | 3,731 | 5,378 |
| 19,769 B Japanese | 7,985 (5,892) | withheld by the guard | 4,942 | 3,469 | 4,451 | 6,415 |
| 13,320 B latin | 13,320 (0) | answered | 3,330 | 3,330 | 3,330 | 3,330 |
| 18,440 B latin | 18,440 (0) | withheld by the guard | 4,610 | 4,610 | 4,610 | 4,610 |

Smaller Japanese reads (3,269 B through 11,963 B) all answered as well, and no size produced the status 255 that a genuine overflow gives. Two things follow. One character per token for CJK is ruled out: 4,668 CJK characters answered, and 5,078 + 331 does not fit 4,096. And four bytes per token is *not* the optimistic figure it looks like for three-byte characters — it is the conservative one, over-counting the same Japanese result that 1.5 characters per token fits inside the window. So `CJK_CHARS_PER_TOKEN` is 1.5, the densest the measurement allows, and `BYTES_PER_TOKEN` stays 4.0; the two then agree within about 1.13x on Japanese and exactly on ASCII, which matters because one number mixes them (prompts and answers as text, tool output as bytes). Making the byte figure script-aware would need the text `ToolCallRecord` does not keep. Long documents are handled by `kennel.context.chunk_text` and `map_reduce` (see `examples/meeting_summary.py`).

## Narration nudge

`Session._should_nudge()` decides whether a turn that only *described* using a tool, instead of calling it, gets one re-prompt (`NUDGE_PROMPT`) telling it to actually call the tool. It stays quiet whenever the turn itself made tool calls, or the *immediately preceding* turn did (`self.history[-1].tool_calls`): either way the model is answering from something it just saw, not narrating unseen steps, and nudging would re-run the same tools and double the answer (#33).

That check does not distinguish a preceding turn that succeeded from one that failed (`stop_reason == "error"`, the shape `_fail_turn` records — same tool calls, empty response). This is deliberate (#46): a failed turn's tool calls are exactly the ones most likely to have overflowed the context window, so nudging into a retry of them is more likely to reproduce the failure than to help. The narration is left alone rather than re-run.

## Events and hooks

`EventBus` delivers `session.*`, `model.*`, `tool.*`, `permission.*` and `context.compacted` events synchronously to subscribers, possibly from a worker thread. The list of types is `EventType` in `src/kennel/events.py`. Events carry summaries, sizes and timings, never file contents; a subscriber that raises is logged and ignored. Consumers: the CLI renderer, `--trace` (raw events on stderr), `--output-format stream-json` (documented schema in [output-format.md](output-format.md)) and `Session.stream()`. A fifth consumer is `JsonlEventLog`, the subscriber that appends the flat `event_json` form to `<state_dir>/sessions/<session_id>.jsonl`; `Session` attaches one with the first turn when `config.log_events` is set and closes it after the `session.completed` event, so a session that is never run leaves no file. The file is `0600` in directories created `0700`, opened through `kennel._statefiles.open_private` — the same opener `FileSessionStore` uses — and the id in its name passes `check_session_id`, which `Session.__init__` applies to every id a caller chooses, with or without a store (#83). The CLI turns it on by default and the SDK does not, because an embedded runtime should not start writing to the user's home on its own; `state_dir()` resolves `KENNEL_STATE_DIR`, then `XDG_STATE_HOME`, then `~/.local/state`. Being an observer, a log that cannot be opened or written warns once through `logging` and then does nothing — the turn is never affected.

Saved conversations are a different file for a different job (#62). `SessionStore` (`src/kennel/sessions.py`) is the restorable record, so it holds text — prompts, answers, stop reasons and each tool call's name and outcome, never arguments, summaries or output — and that is why it is opt-in everywhere: an `Agent` saves only when it is given a `session_store`, and the CLI passes a `FileSessionStore` only for `--persist` / `sessions.persist` (or, read-only, for `--continue` / `--resume`). `FileSessionStore` keeps `<state_dir>/transcripts/<sha256(workspace root)[:16]>/<session_id>.jsonl`, apart from the event log's `sessions/`. `Session` claims the id with the store's `lock` when it is created (non-blocking `flock`, released by `close()`), fills `history` from `load` when an id was passed, and appends every turn through `Session._record` — the one place both `run()` and `_fail_turn()` add to the history — where a store failure is logged, not raised. Resuming restores only the history: no provider session exists yet, so the first turn opens one seeded with `_compaction_note()`, exactly as after `compact()`. Session grants stay in the agent's `PermissionManager` and die with the process.

Hooks (`src/kennel/hooks.py`) are the opposite kind of callback: they decide. `before_tool` / `after_tool` run inside `ToolRunner.invoke` and a hook that raises fails the call; `before_prompt` runs in `Session.run` and appends context to the user's prompt (not to Kennel's own re-prompts). They execute in-process with the application's own trust level, see [SECURITY.md](../SECURITY.md).

## Providers

`ModelProvider.create_session()` returns a `ProviderSession` with `respond`, `stream`, optional `respond_structured` and `usage`, and `close`. `ProviderInfo` declares `name`, `model`, `mode` (`local` means inference and data stay on this machine; anything else must say so) and `context_window_tokens`. `MockProvider` scripts turns, tool calls, structured values and reported usage for the test suite (`tests/providers/`).

`mode` is not a setting: `LlamaServerProvider` derives it from its `base_url`, and only a loopback host (`127.0.0.0/8`, `::1`, `localhost`) may call itself `local`. `LlamaCppProvider` is `local` unconditionally, which it has earned: the model is in this process and no port is opened at all. It is decided without any I/O, at construction, so a remote endpoint shows up in the CLI header before the first request. The provider talks to a server the user started: Kennel neither downloads a model nor launches a process, and adds nothing to a request body unless `extra_body` says so. `check_availability()` is where `/props` (`n_ctx`) and `/v1/models` (the model name) are read, and it runs once on the first `create_session()` if the caller never asked, so `info` is honest for an SDK consumer too. `LlamaCppProvider` answers the same question the same way and deliberately *without* loading anything: the import, the file and `n_ctx`, so `kennel doctor` and `Agent(...)` never wait for a model. `create_session()` calls it before the load for the same reason llama-server probes there — an SDK caller who never asked still gets the install hint or the path, not an `ImportError` from inside the load.

Providers are chosen by name, and `kennel.providers.registry` is the only place that maps a
name to an implementation: a static dict of `ProviderSpec(name, factory, doctor_checks, secrets)`
plus `register()`, shared by `Agent`, the CLI and `kennel doctor` (entry points are deliberately
not read). `factory(**options)` takes provider-specific keyword arguments; `doctor_checks(**options)`
is optional and returns the `Check`s (`kennel.diagnostics`) that `kennel doctor` shows for that
provider — the `apple_fm_sdk` and model-availability checks live with `AppleProvider`.
`secrets` (last, so the positional form keeps its meaning) declares which factory arguments are
API keys, with the same `Secret` and `kennel.credentials` functions as search providers:
`resolve_options(name, options)` is the one place that refuses a key written as a config value,
reads each declared key from its environment variable and returns the factory's arguments, and
both `create()` and `kennel doctor` go through it (the config loader runs the same refusal when
it reads `providers.<name>`). Only `llama-server` declares one, the optional `api_key` from
`KENNEL_LLAMA_SERVER_API_KEY`; the provider gets the value, never the variable's name, so its
HTTP 401 message says only whether a key was sent, and `kennel doctor` names the variable.

```python
from kennel import ProviderSpec, register_provider, create_provider

register_provider(ProviderSpec("echo", lambda **options: EchoProvider(**options)))
provider = create_provider("echo")
```

The selection is configured with two keys, `provider` (a name) and `providers` (options per
name), so switching with `--provider <name>` keeps each provider's options where they are:

```json
{ "provider": "mock", "providers": { "mock": { "script": "{\"turns\": [\"hi\"]}" } } }
```

`Agent(provider=)` takes a `ModelProvider` instance, a name, or `None` (use `provider` from the
config, default `apple`); an instance always wins. Precedence is the usual one —
`--provider` > `Agent(provider=)` > `./kennel.json` > user config > `apple` — and it is
implemented once, in `KennelConfig.merged()`. Choosing `apple` where `apple_fm_sdk` cannot be
imported fails immediately with `ModelUnavailableError`, whose hint points at `--provider`.

## Web search

The `web` tool calls a `SearchProvider` (`kennel.search`): anything with an `info`
(`SearchProviderInfo(name, destination, mode)`) and an async `search()`. An empty list means
the service looked and found nothing; a search that could not be carried out raises
`SearchError`, whose *message* says whether retrying can help and what to fix, because the
model only ever receives `Error: <message>` from `ToolRunner._fail`. `mode` is the search
service's own, `remote` unless the query never leaves the machine; it is shown on the header's
`web search:` line and in `Session.status()["web"]`, never folded into the model's `mode`.

Search providers are chosen by name in `kennel.search.registry`, the same static dict plus
`register()` as model providers. `SearchSpec.secrets` declares which factory arguments are
secrets (`{"api_key": Secret("BRAVE_API_KEY")}`); `kennel.credentials` reads them from the
environment, refuses them as config values, and never puts a value in a message, so a provider
does not know where its key came from. A missing key does not stop `Agent(...)`: the tool is
built "unavailable" and names the variable when called. Turning the `search_provider` name into
a tool happens only in `ToolRegistry`, whose `web` factory is registered `configured=True` and
receives the effective config; an explicit `WebSearchTool(provider)` instance is used as given.

The built-in providers (`searxng`, `brave`) only build a request and parse the answer. The
HTTP GET is `kennel.search.http.http_get`, a thin wrapper over `kennel._http.bounded_request`:
`http.client` on a worker thread, a deadline for the whole request (not only per socket
operation), a socket shutdown when the awaiting task is cancelled (`Session.interrupt()`), and
a byte limit at which reading stops. `llama_server.py` has its own streaming client; the two
were not merged because one streams SSE and the other reads a bounded body.

## Fetch

The `fetch` tool (`kennel.tools.fetch`) uses the same `bounded_request`, with its own way of
opening the connection: on the worker thread it resolves the name, refuses the request if any
address is not global, and connects to a checked address itself (non-blocking, polling the
cancel token every 50 ms), wrapping TLS with the original name for SNI and certificate checks;
`http.client` gets the connected socket and never resolves the name again. `getaddrinfo`
cannot be interrupted, so a cancel during resolution abandons the worker, which exits without
connecting when the lookup returns; the check before sending and the cancel take the same
lock, so nothing is sent after a cancel. Redirects are followed by the tool, one
`bounded_request` per hop, within one overall deadline.

Rules match the normalised host (`FetchTool.match_rule`), and "allow for this session" is
remembered per host through `Tool.session_scope`: `PermissionManager` stores grants as
`(tool name, scope)`, the runner passes the scope with the `PermissionRequest` and, once the
call is certain to run, grants the scope of its final arguments (step 4 above). The default scope
`None` is the whole tool, which every other tool keeps.
