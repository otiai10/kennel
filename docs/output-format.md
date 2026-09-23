# Machine-readable output (`--output-format`, `--json-schema`)

Schema version: **1**. This document is the contract for anything that parses Kennel's
stdout. Adding a key is not a breaking change; removing or retyping one is, and bumps the
schema version (noted here and in the release notes). Parsers should ignore unknown keys.

`--output-format` only applies to a one-shot run, so it requires `-p/--prompt`:

```bash
kennel . -p "summarize the README" --output-format json
kennel . -p "summarize the README" --output-format stream-json
```

| Value | stdout |
| --- | --- |
| `text` (default) | human output: one line per tool call, then the streamed answer — unless `--json-schema` is given, in which case stdout is the structured JSON document alone |
| `json` | exactly one line: the [result object](#result-object) |
| `stream-json` | one JSON object per line ([events](#event-objects)), the result object last |

`--output-format json|stream-json` (and `--json-schema`) without `-p` exits **2** with a
message on stderr; that error is plain text, because the output contract is only established
once the flags are valid.

In `json` and `stream-json` mode **stdout carries nothing but JSON**: tool lines, the
streamed answer and notes are suppressed. Everything human moves to stderr — `error: ...`
messages, `--verbose` logging, the interactive permission prompt, and `--trace` (which keeps
writing its own event lines to stderr and can be combined with either format).

## `--json-schema`

`--json-schema TEXT|@FILE` answers under guided generation instead of prose. The value is a
JSON schema, either inline or read from a file named after a leading `@`; anything that is
not a JSON **object** is rejected with exit **2**. Tools still work — the provider calls them
inside the same request — so `tool_calls` is populated as usual.

```bash
kennel . -p "extract the decisions" --json-schema @schema.json
# { "title": "Release planning", "decisions": ["Ship v0.1 on Friday"] }

kennel . -p "extract the decisions" --json-schema @schema.json --output-format json
# {"type":"result","text":"{\n  \"title\": ...}","structured_output":{"title":"Release planning", ...}, ...}
```

With the default `text` format, stdout is the document alone (tool lines are suppressed, the
same rule as the machine formats). With `json` / `stream-json` the document goes into
`structured_output` and its text into `text`; the two always agree, so
`json.loads(result["text"]) == result["structured_output"]`.

## Result object

Emitted last in both machine formats. `AgentResult.to_dict()` in the SDK produces the same
object.

| Key | Type | Meaning |
| --- | --- | --- |
| `type` | `"result"` | discriminator; always `result` |
| `text` | string | the final answer (empty on error) |
| `stop_reason` | `"end_turn"` \| `"tool_limit"` \| `"timeout"` \| `"cancelled"` \| `"error"` | why the turn ended |
| `is_error` | boolean | the turn produced no usable answer. `tool_limit` is **not** an error: the answer is there, only possibly incomplete |
| `error` | string \| null | the user-facing message when `is_error` is true |
| `duration_ms` | number | wall-clock time of the turn (0 when it failed before starting) |
| `session_id` | string | the session this turn ran in (empty when it failed before the session started) |
| `tool_calls` | array of [tool call](#tool-call-object) | every tool call of this turn, in order |
| `usage` | object \| null | `{"input_tokens", "output_tokens"}`; `null` while the on-device provider reports no token counts |
| `structured_output` | object \| null | the guided-generation value when `--json-schema` (SDK: `run(schema=)`) was used, else `null`. When set, `text` is the same document as JSON |
| `compactions` | integer | how often the conversation was compacted to fit the context window |
| `failure` | object \| null | the [failure object](#failure-object) when a turn failed, else `null` |

The exit code is unchanged by the output format: `0` ok, `1` runtime error or turn timeout,
`2` bad configuration or arguments, `3` model unavailable, `130` interrupted. An error that
happens after the flag was accepted is reported **both** as a result object with
`is_error: true` on stdout and as `error: ...` on stderr.

### Tool call object

The fields of `kennel.ToolCallRecord`:

| Key | Type | Meaning |
| --- | --- | --- |
| `name` | string | tool name |
| `arguments` | object | validated arguments |
| `summary` | string | one-line human summary (`Read README.md [1-50]`); for `fetch` the URL without its query and fragment (`Fetch https://docs.python.org/3/library/`), while `arguments` keeps the whole URL |
| `status` | `"ok"` \| `"error"` \| `"denied"` \| `"blocked"` \| `"invalid"` | outcome |
| `output_bytes` | integer | size of the result the tool produced (what the model received, unless `withheld`) |
| `truncated` | boolean | the result was cut to the output limit |
| `duration_ms` | number | execution time |
| `error` | string \| null | why it failed |
| `metadata` | object | tool-specific extras; counts and names, never content (`web`: `{"provider", "returned"}`, the search provider's name and the number of results; `fetch`: `{"host", "status", "bytes", "content_type"}`, the normalised host, the HTTP status, the size of the body received and its media type, never the page) |
| `withheld` | boolean | the result was too big for the model's whole context window, so the model got a short "ask for a smaller part" instruction instead of it |

### Failure object

What Kennel knows about the turn that failed. The same object is the `data` of the
`session.failed` event and is reachable from the SDK as `Session.last_failure`, so a
consumer of either gets the same facts. Every number is counted or reported by the
provider; nothing here is inferred (constitution principle 4).

| Key | Type | Meaning |
| --- | --- | --- |
| `error` | string | the user-facing message |
| `error_type` | string | the Kennel error class (`ProviderError`, `ContextLimitError`, ...) |
| `status` | integer \| null | the status code the provider reported, when it reports one (Apple's `255` is `GenerationErrorCode.UNKNOWN_ERROR`) |
| `provider_error` | string \| null | the provider's own wording, when Kennel replaced it with a friendlier message |
| `tool_output_bytes` | integer | how many bytes of tool output this turn produced, withheld results included |
| `tool_calls` | integer | how many tool calls this turn made |
| `context_tokens_before` | integer | how full the window was **before** the request |
| `context_window_tokens` | integer \| null | the window the provider declares, `null` when it declares none |
| `estimated` | boolean | whether `context_tokens_before` is an estimate (true while the provider counts no tokens) |
| `duration_ms` | number | how long the turn ran before it failed |

A failed turn is also kept in the session's history with `stop_reason: "error"` and the tool
calls it did make, so `/usage`, `/status` and `Session.context_usage()` account for it. The
provider session is retired after a failure (the conversation is compacted), because the
provider gives no way to tell whether the failed prompt stayed in its transcript;
`compactions` therefore grows by one and the next turn resumes from a summary.

## Event objects

`stream-json` only. One line per event, in the order the agent emitted them:

```json
{"type": "tool.completed", "session_id": "2f1c9a4b7e03", "timestamp": 1789531200.42, "data": {"tool": "read", "summary": "Read README.md", "output_bytes": 1024, "estimated_tokens": 256, "context_window_tokens": 4096, "window_exceeded": false, "withheld": false, "truncated": false, "duration_ms": 3.1, "metadata": {}}}
```

| Key | Type | Meaning |
| --- | --- | --- |
| `type` | string | event type (below) |
| `session_id` | string | the session that emitted it |
| `timestamp` | number | Unix seconds, milliseconds precision |
| `data` | object | per-type payload; keys vary and may grow |

Event types: `session.started`, `session.completed`, `session.failed`, `session.cancelled`,
`model.started`, `model.delta`, `model.completed`, `model.nudged`, `tool.requested`,
`permission.requested`, `permission.denied`, `tool.started`, `tool.completed`,
`tool.failed`, `tool.blocked`, `context.compacted`. `session.cancelled` follows an
interrupted turn and `tool.blocked` a call a `before_tool` hook denied; both are the
`EventType` members of `kennel.events`, which is the authoritative list.

`session.failed` carries the [failure object](#failure-object) as its `data`, and is
preceded by the `context.compacted` of retiring the provider session.

`tool.completed` says how big the result handed to the model was, in bytes and in estimated
tokens:

| Key | Type | Meaning |
| --- | --- | --- |
| `output_bytes` | integer | size of the result handed to the model |
| `estimated_tokens` | integer | what those bytes are estimated to cost. An estimate, as the name says: the on-device provider counts no tokens (constitution principle 4) |
| `context_window_tokens` | integer \| null | the window the provider declares, `null` when it declares none |
| `window_exceeded` | boolean | `estimated_tokens > context_window_tokens` — **this one result** is already too big for the window, whatever else the conversation holds. It is not a statement about how full the context currently is; for that, use `Session.context_usage()`. Always `false` when the provider declares no window |
| `withheld` | boolean | what the runner did about it: the result was **not** handed to the model, which got a short instruction to ask for a smaller part (a line range, a grep) instead. `window_exceeded` is the observation, `withheld` the intervention; today the second follows the first, and the sizes above keep describing the result the tool produced, not the notice |

A withheld result costs the model nothing but the notice, so `Session.context_usage()` does
not count it. The output limit is untouched by all this: `tools.max_output_bytes` (default
65,536) still bounds every result, and setting it below the window is how a caller goes back
to receiving truncated content instead of the notice.

The CLI draws this line for every tool call whether or not `--verbose` is set
(`↳ 20,008 bytes (~5,002 tokens, exceeds the 4,096-token window, withheld)`, in yellow when
the window is exceeded), and prints `(context: ...)` after each turn. `--verbose` only adds
the timing. Neither appears in `json` / `stream-json` mode, where stdout is JSON alone.

`model.delta` carries `{"text": "..."}` — the generated fragment. Guided generation arrives
whole, so a `--json-schema` turn emits exactly one `model.delta` holding the document. Every
other event carries summaries, sizes and durations only, never file contents or generated text: that invariant
belongs to the event bus (`kennel.events`) and `--trace` relies on it too. Concatenating the
`model.delta` texts reproduces `result.text`.

A consumer that only wants the outcome can read the last line; one that wants progress can
switch on `type` and ignore the rest.
