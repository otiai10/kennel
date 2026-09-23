# Kennel

**Kennel is a lightweight local agent runtime and Python SDK for Apple Foundation Models.**

It gives Apple's on-device model a small set of local tools (`glob`, `grep`, `read`, and
permission-gated `write`, `edit`, `shell`), a workspace boundary, a permission layer, an
event stream, and a CLI. Everything runs on your Mac; nothing leaves it by default.

```text
$ kennel ~/meetings

Kennel v0.1.0
workspace: /Users/me/meetings
provider: apple
model: Apple SystemLanguageModel
mode: local
permissions: default
tools: glob, grep, read, write (ask), edit (ask), shell (ask)

> 最新の会議文字起こしを探して、決定事項とTODOをまとめて

● Glob **/*.txt
● Read transcripts/2026-09-16.txt [1-50]

決定事項: v0.1 を金曜日にリリース ...
```

Kennel is not a Claude Code clone. It is the thin layer between an application (or the
terminal) and the `apple_fm_sdk` tool-calling loop: workspace management, path safety,
permissions, bounded tool output, context handling, tracing, and a small CLI.

## Requirements

- macOS 26+ on Apple Silicon (developed and tested on macOS 27.0)
- Apple Intelligence turned on (System Settings > Apple Intelligence & Siri)
- Xcode 26+ installed, with the license agreed (the SDK builds a Swift bridge on install)
- Python 3.10+

The first three are what the default `apple` provider needs. Kennel itself installs on other
platforms too (`apple-fm-sdk` carries a `sys_platform == "darwin"` marker), where you pick
another provider with `--provider` or the `provider` config key.

## Install

Kennel is not on PyPI yet (the name is pending a [PEP 541 request](https://github.com/pypi/support/issues/12302)).
Install straight from GitHub. The recommended way is an isolated tool install with [uv](https://docs.astral.sh/uv/)
or pipx, which puts a `kennel` command on your PATH:

```bash
uv tool install "git+https://github.com/otiai10/kennel"
# or
pipx install "git+https://github.com/otiai10/kennel"
kennel --help
```

To try it without installing:

```bash
uvx --from "git+https://github.com/otiai10/kennel" kennel --help
```

Plain pip into an existing environment works too. Pin a tag or branch with `@v0.1.0` at the end of the URL.

```bash
pip install "git+https://github.com/otiai10/kennel"
```

For development, use an editable install:

```bash
git clone https://github.com/otiai10/kennel && cd kennel
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"    # or: uv sync --extra dev
kennel --help
```

There is one optional extra, `llama-cpp`, for the in-process llama.cpp provider
([below](#llama-cpp-the-same-model-in-this-process-no-server-no-port)). It is not installed
by default, and PyPI ships an sdist only, so it builds llama.cpp from source: about a minute
on Apple Silicon, Xcode Command Line Tools required, cmake is fetched automatically.

```bash
pip install -e ".[dev,llama-cpp]"    # or: uv sync --extra dev --extra llama-cpp
```

> **About the `apple-fm-sdk<0.2.1` pin.** The SDK has no wheels; installing it builds a Swift
> bridge with your Xcode. Version 0.2.1 references APIs that only exist in the macOS 27 SDK
> shipped with Xcode 27, so with Xcode 26.x it fails to compile. The upper bound will be
> lifted once a matching Xcode is common. The dependency is marked
> `; sys_platform == "darwin"`: on a Mac `uv tool install` still brings the SDK along, and on
> Linux the `kennel` command installs without any Swift build.

Check that Kennel can run here:

```bash
kennel doctor
```

This checks the Python version, platform, the selected provider (for `apple`: the `apple_fm_sdk`
install and model availability), the web search service if one is configured (see
[Web search](#web-search-off-by-default-no-default-service)), Xcode, user/project config files,
the workspace, and prints the effective tools/permissions.
Failing checks show `✗` with a reason and, where there is one, a fix; add `--json` for a
machine-readable report (handy when filing a bug: paste the output of `kennel doctor --json`).
Exit code is `0` when everything checks out, `1` otherwise.

## CLI

```bash
kennel                      # interactive session in the current directory
kennel ~/meetings           # interactive session in another workspace
kennel -p "Read the README and explain this project"   # one-shot
```

| Option | Effect |
| --- | --- |
| `-p, --prompt TEXT` | run one prompt, print the answer, exit |
| `--permission-mode MODE` | `read-only`, `default`, `accept-edits`, `dont-ask` or `bypass` |
| `--read-only` | only `glob`, `grep`, `read` are available (= `--permission-mode read-only`) |
| `--allow-write` | `write` and `edit` run without asking (= `--permission-mode accept-edits`) |
| `--allow-shell` | `shell` runs without asking (see Security) |
| `--allow-web` | enable the `web` tool (asks; needs a configured search provider, see [Web search](#web-search-off-by-default-no-default-service)) |
| `--allow-fetch` | enable the `fetch` tool (asks, per host; see [Fetch](#fetch-off-by-default-public-addresses-only)) |
| `--non-interactive` | never prompt; anything that would ask is denied (= `--permission-mode dont-ask`) |
| `--instructions TEXT\|@FILE` | append text (or a file's contents) to the default instructions |
| `--system-prompt TEXT\|@FILE` | replace the default instructions entirely (or a file's contents) |
| `--max-tool-calls N` | tool call budget per turn (default 32) |
| `--provider NAME` | model provider to use (default: the `provider` config key, otherwise `apple`) |
| `--output-format FORMAT` | with `-p`: `text` (default), `json`, `stream-json` |
| `--json-schema TEXT\|@FILE` | with `-p`: answer under a JSON schema (guided generation) |
| `--verbose` | add tool timings and diagnostic logging (result sizes are shown anyway) |
| `--trace` | write every agent event as JSON lines to stderr |
| `--no-log` | do not keep this session's event log (= `"logging": {"events": false}`) |
| `--persist` / `--no-persist` | save this conversation so it can be resumed, or not, whatever the config says (= `"sessions": {"persist": true}`; off by default) |
| `--continue` | resume the most recently saved conversation of this workspace |
| `--resume ID` | resume the saved conversation `ID` of this workspace |

Interactive commands: `/help`, `/status`, `/usage`, `/clear`, `/compact`, `/permissions [<tool>
allow|ask|deny]`, `/exit`. `/status` includes the path of this session's event
log and whether the session was resumed. `Ctrl-C` cancels the current answer via `Session.interrupt()` and
returns to the prompt (twice at the prompt exits); `Ctrl-D` exits.

### Resuming a conversation

Saving conversations is **off by default**: a saved conversation keeps the text of your
prompts and of the answers on disk. Turn it on once with `"sessions": {"persist": true}` in
`~/.config/kennel/settings.json` (or `./kennel.json`), or per run with `--persist`;
`--no-persist` wins over the config. One-shot runs (`-p`) are saved the same way as
interactive ones.

```bash
kennel --persist ~/meetings          # talk, then /exit
kennel --continue ~/meetings         # the most recent saved conversation of this workspace
kennel --resume 06c84876ce05 ~/meetings   # a given one (the session_id from /status or the JSON result)
```

A saved conversation lives in `<state dir>/transcripts/<workspace hash>/<session_id>.jsonl`
(the state dir is the one the event log uses: `KENNEL_STATE_DIR`, else
`$XDG_STATE_HOME/kennel`, else `~/.local/state/kennel`), in a `0700` directory as a `0600`
file. Each line is one turn: the prompt, the answer, why it stopped and, for each tool call,
only the tool's name and outcome — never its arguments, command line or output.
`--continue` and `--resume` only look at the conversations of the workspace you open, and
they read a saved conversation even when saving is off (the resumed turns are then not
written back). A session that is open in one `kennel` cannot be opened in a second one.

The model does not get the old session back — Apple's session cannot be saved — but a
summary of the saved turns, the same one `/compact` builds. "Allow for this session"
approvals are not saved: a resumed session asks again. `/clear` deletes the saved
conversation (keeping the id; the next turn starts a new file), and says so if it could not.

The on-device model's context window is about 4k tokens, which is the tightest constraint in
practice, so every tool result is reported with what it costs and every answer ends with how
full the window is. A result that cannot fit the window on its own is not handed to the model
at all — it would fail the request rather than shorten it — so Kennel says so in yellow and
tells the model to ask for a smaller part instead:

```text
> read big.txt and summarize it
● Read big.txt
  ↳ 20,008 bytes (~5,002 tokens, exceeds the 4,096-token window, withheld)
● Read big.txt [1-200]
  ↳ 4,912 bytes (~1,228 tokens)
big.txt starts with ...
(context: 38% of 4096 tokens (1556 tokens, estimated: the provider reports no token counts))
```

Only the whole-window case is withheld; everything else is bounded by `tools.max_output_bytes`
(64KB by default) as before. Lowering that setting below the window is how you go back to
receiving truncated content instead of the notice.

The token figures are estimates and say so; the on-device model exposes no token counter, and
Kennel does not dress an estimate up as a measurement. `/usage` is the longer form:

```text
> /usage
window: 4096 tokens
used: 504 tokens (estimated: the provider reports no token counts)
context: 12%
turns: 1
compactions: 0
```

Both lines are human output, so `--output-format json|stream-json` suppresses them (there the
same numbers are on the `tool.completed` records). `--verbose` only adds the timing.

For scripts and other processes, `-p` can print machine-readable JSON instead of the human
rendering:

```bash
kennel . -p "summarize the README" --output-format json
# {"type":"result","text":"...","stop_reason":"end_turn","is_error":false,"duration_ms":1234, ...}

kennel . -p "summarize the README" --output-format stream-json
# {"type":"session.started","session_id":"...","timestamp":...,"data":{...}}
# {"type":"model.delta","session_id":"...","timestamp":...,"data":{"text":"..."}}
# {"type":"result", ...}
```

In both formats stdout is JSON only; diagnostics stay on stderr. The keys, the event types
and the versioning promise are documented in [docs/output-format.md](docs/output-format.md).

`--json-schema` answers under a schema instead of in prose. Tools still work, so the model
can look things up and then fill the schema in one request:

```bash
kennel ~/meetings -p "extract the decisions from the latest transcript" --json-schema @schema.json
# {"title": "Release planning", "decisions": ["Ship v0.1 on Friday"]}
```

The three older flags are sugar for a mode, so `--permission-mode` cannot be combined with
them. `bypass` allows everything including `shell`, `web` and `fetch`, and prints a warning line in the header.
What each mode allows, tool by tool, is tabulated in [SECURITY.md](SECURITY.md#what-is-enforced).

`/compact` summarizes the conversation so far and starts a fresh model session seeded with
that summary (this happens automatically when a turn no longer fits the context window;
`/compact` lets you do it on your own terms). `/permissions` alone prints the decision
(`allow`/`ask`/`deny`) and session grant for every enabled tool (for `fetch`, the hosts granted); `/permissions shell allow`
changes a tool's decision for the rest of the session (not written back to `kennel.json`).

Every tool call is shown as one line (`● Read transcripts/2026-09-16.txt [1-50]`). Mutations
ask first:

```text
? Allow Write: Write minutes.md
  Creates new file minutes.md (312 bytes, 14 lines)
  [y] once  [a] session  [n] deny >
```

## Python SDK

```python
import asyncio
from kennel import Agent

async def main():
    agent = Agent(workspace="~/meetings", tools=["glob", "grep", "read"])
    agent.check_availability()
    result = await agent.run("Find the latest transcript and list the decisions")
    print(result.text)
    for call in result.tool_calls:
        print(call.summary, call.status)

asyncio.run(main())
```

`Agent.run()` returns an `AgentResult` (`text`, `stop_reason`, `tool_calls`, `usage`,
`session_id`, `duration_ms`, `is_error`, `structured_output`, `compactions`, `failure`). `usage` is a
`Usage` (`input_tokens`, `output_tokens`) and is only filled when the provider counts tokens
itself; on Apple's on-device model it stays `None`, because the SDK exposes no token counter.
Kennel does not guess it per turn — use `context_usage()` below for a picture of the window.
`result.to_dict()` gives the same JSON object the CLI prints with `--output-format json`.
Multi-turn conversations use a session:

```python
session = agent.new_session()
await session.run("Summarize transcripts/2026-09-16.txt")
await session.run("Now only the TODOs")
```

To keep a conversation across processes, give the agent a store; `new_session(session_id=...)`
then reads that conversation back (`session.resumed` says whether it found one). Without a
store nothing is saved:

```python
from kennel import Agent, FileSessionStore

agent = Agent("~/meetings", session_store=FileSessionStore("~/meetings"))
session = agent.new_session()                       # every turn is appended as it lands
...
later = agent.new_session(session_id=session.id)    # another process, the next day
```

`FileSessionStore` is one implementation of the `SessionStore` protocol (`append`, `load`,
`latest`, `clear`, `lock`) — see [Resuming a conversation](#resuming-a-conversation) for what
it keeps and where.

To follow a turn as it happens, iterate it instead. `Session.stream()` yields the session's
events on *your* event loop — including the ones a provider fires from a worker thread — so
no queue plumbing or thread-safety work is needed on the application side. A session is also
an async context manager, so `close()` runs on the way out:

```python
async with agent.new_session() as session:
    async for event in session.stream("Summarize the latest transcript"):
        if event.type == "model.delta":
            print(event.data["text"], end="", flush=True)
        elif event.type.startswith("tool."):
            print(event.type, event.data.get("summary"))
    print(session.last_result.stop_reason)      # the full AgentResult of the last turn
```

The iterator's last event is `session.completed`, whose `data` carries `text`, `stop_reason`,
`tool_calls` and `turns`. A provider failure is raised out of the `async for` after the
`session.failed` event has been delivered; that event's `data` (also `Session.last_failure`,
and `AgentResult.failure` in the machine formats) says what the failed turn had sent — the
provider's status code, the bytes of tool output it added and how full the window was before
the request, flagged `estimated` while the provider counts no tokens. The failed turn stays in
the history with `stop_reason "error"`. If the request had already gone to the provider, its
provider session is retired, so the next turn resumes from a summary; a failure before
anything was sent (a raising `before_prompt` hook) keeps the session as it was. `Agent.stream(prompt)` is the one-shot form: it
runs a throwaway session and its final `session.completed` event carries the same `text` that
`Agent.run()` would return. `run(on_delta=...)` still works and is unchanged.

A running turn can be stopped from anywhere — another task, another thread, a GUI's stop
button — with `Session.interrupt()`. The waiting `run()` emits `session.cancelled` and
raises `TurnCancelledError`; the session stays usable for the next turn:

```python
from kennel import TurnCancelledError

task = asyncio.create_task(session.run("Summarize every transcript"))
session.interrupt()                    # safe from any thread or task; a no-op when idle
try:
    await task
except TurnCancelledError:
    print("stopped")
await session.run("Just the latest one, then")   # the session is still good
```

A plain `task.cancel()` is left alone and still surfaces as `asyncio.CancelledError`, so
cancellation coming from an outer scope keeps asyncio's meaning. The CLI's `Ctrl-C` goes
through `interrupt()` too, so the terminal and an embedding application take the same path.

Because the window is small, `Session.context_usage()` tells you how much of it is left
before you decide how big the next chunk should be:

```python
u = session.context_usage()
u.window_tokens     # 4096 on Apple's on-device model; None if the provider declares none
u.used_tokens       # what the live provider session holds right now
u.ratio             # 0.0-1.0 (0.0 when no window is declared)
u.estimated         # True when Kennel estimated it from text rather than being told
u.estimate_reason   # why: None (measured), "not_reported" or "no_live_session"
u.turns, u.compactions
print(u.summary())  # "12% of 4096 tokens (504 tokens, estimated: the provider reports no token counts)"
```

`used_tokens` counts what is *in the live provider session*: the instructions plus the turns
since it was opened. Compacting the conversation replaces that session with a summarized one,
so the number drops — which is what makes it useful as a budget. A provider that counts tokens
itself can implement `ProviderSession.usage()`; its figure is then used for both
`context_usage()` (with `estimated` False) and `AgentResult.usage`.

Permissions are rules mapped to `allow`, `ask` or `deny`; `ask` needs a prompter, otherwise
it means `deny`. A rule is a tool name, or a tool name with a specifier the tool interprets —
the command line for `shell`, the workspace-relative path for the file tools, and the joined
argument values for a custom tool:

```python
from kennel import Agent, Approval

def prompter(request):            # request.summary, request.details (diff / command),
    return Approval.ONCE          # request.warnings, request.arguments

agent = Agent(".", tools=["glob", "grep", "read", "write", "edit", "shell"],
              permission_mode="accept-edits",     # the defaults rules are layered on
              permissions={"shell": "ask",
                           "shell(git *)": "allow",
                           "shell(git push*)": "deny",
                           "write(docs/**)": "allow",
                           "read(**/.env)": "deny"},
              prompter=prompter)
```

`deny` always wins, whether it came from `shell` or from `shell(rm *)`. Otherwise a matching
specifier rule beats the bare tool rule, and `ask` beats `allow` among equally specific
matches. `permission_mode` is one of `read-only`, `default`, `accept-edits`, `dont-ask`,
`bypass` (see the table in [SECURITY.md](SECURITY.md)); `read-only` also restricts the tool set.

### Hooks

Events report what happened; hooks decide what happens. An application embedding Kennel can
stop a tool call, correct its arguments, replace a result, or add context to a prompt —
without subclassing anything. Callbacks may be sync or async:

```python
from kennel import Agent, Allow, Deny, HookMatcher, Hooks, ToolResult

async def guard(call, ctx):                    # before_tool: runs before the permission check
    if "curl" in call.arguments.get("command", ""):
        return Deny("network access is not allowed in this workspace")
    return None                                # None means "no opinion"

def cap_reads(call, ctx):
    return Allow(updated_arguments={**call.arguments, "end_line": 100})

def redact(call, result, ctx):                 # after_tool: return a ToolResult to replace it
    return ToolResult(result.content.replace(SECRET, "***"))

def add_context(prompt, session):              # before_prompt: appended to what the model sees
    return "Today is 2026-09-18."

agent = Agent(".", hooks=Hooks(
    before_tool=[guard, HookMatcher(tools=["read"], hooks=[cap_reads])],
    after_tool=[redact],
    before_prompt=[add_context],
))
agent.hooks.before_tool.append(another_guard)   # also fine after construction
```

- `before_tool(call, ctx)` runs after the arguments are validated and **before** the
  permission check, so a policy can deny what the permission policy would have allowed.
  `Deny` records the call as `blocked`, emits `tool.blocked` and returns the message to the
  model; `Allow(updated_arguments=...)` re-validates and continues; `Allow(remember="session")`
  grants the tool for the session.
- `after_tool(call, result, ctx)` may return a `ToolResult` to replace the result. The
  replacement is bounded by the same output limit.
- `before_prompt(prompt, session)` returns text appended to the prompt sent to the model.
  The conversation history keeps the user's own words, and Kennel's internal re-prompts
  (the narration guard) are not passed through the hook.
- `HookMatcher(tools=[...], hooks=[...])` restricts hooks to named tools.
- A hook that raises fails the tool call (`status="error"`). A broken policy is an error,
  not an absence of policy — unlike an event subscriber, whose exceptions are only logged.
  A `before_prompt` hook has no tool call to fail, so it fails the turn instead: `run()`
  raises `HookError` (a `KennelError`, so the CLI prints `error: hook failed: ...` and
  stays in its prompt) after emitting `session.failed`.
- `ctx` is a `HookContext` carrying the `session_id`, the `tool` and the same `ToolContext`
  the tool will run with (`ctx.workspace`, `ctx.permissions`, `ctx.environment`). Hooks run
  on the provider's tool thread, so they get plain data and thread-safe services only, and
  they must not hold anything bound to another event loop.

The prompter can answer with more than an `Approval` too:

```python
def prompter(request):
    if request.tool_name == "shell":
        return Deny(message="shell is disabled for this workspace; use the file tools")
    return Allow(remember="session")           # or Approval.ONCE / SESSION / DENY
```

`Deny.message` is what the model is told, so make it actionable; `Allow(updated_arguments=)`
corrects the call before it runs.

Custom tools subclass `kennel.Tool` and are passed alongside built-in names:

```python
from kennel import Tool, ToolParameter, ToolResult, PermissionKind

class ListMeetings(Tool):
    name = "list_meetings"
    description = "List meetings known to the app."
    permission = PermissionKind.READ
    parameters = (ToolParameter("limit", "integer", "How many", required=False),)

    async def execute(self, arguments, context):
        return ToolResult("2026-09-16 release planning\n2026-09-15 weekly sync")

agent = Agent(".", tools=["read", ListMeetings()])
```

Events (`session.started`, `session.completed`, `session.failed`, `session.cancelled`,
`model.started`, `model.delta`, `model.completed`, `model.nudged`, `tool.requested`,
`tool.started`, `tool.completed`, `tool.failed`, `tool.blocked`, `permission.requested`,
`permission.denied`, `context.compacted`) are available through
`agent.events.subscribe(callback)`; the CLI renderer is just one subscriber. Events carry
summaries and sizes, never file contents. Subscribers only observe — to intervene, use
hooks (above). `Session.stream()` is the async-iterator view of the same events, for
consumers that would rather `async for` than register a callback.

### Session event log

Every `kennel` run keeps its events in a local JSON Lines file, so a failure can be looked at
after the fact instead of only while it happens:

```
~/.local/state/kennel/sessions/<session_id>.jsonl
{"t": 1789817185.923, "type": "session.started", "session_id": "06c84876ce05", "workspace": "/Users/me/meetings", "model": "MockModel", "tools": ["glob", "grep", "read"]}
{"t": 1789817185.924, "type": "tool.requested", "session_id": "06c84876ce05", "tool": "read", "summary": "Read transcripts/2026-09-16.txt"}
```

Each line is one event in the same flat form `--trace` writes to stderr (`t`, `type`,
`session_id`, then the event's own data), so it carries summaries, sizes and timings and never
file contents or generated text. The location is `KENNEL_STATE_DIR`, else
`$XDG_STATE_HOME/kennel`, else `~/.local/state/kennel`; `/status` and `kennel doctor` print
it. `--no-log` or `"logging": {"events": false}` turns it off. Nothing is uploaded and nothing
is rotated for you — the files are yours to `tail`, attach to an issue or delete.

In the SDK the log is **opt-in**, so embedding Kennel does not start writing files behind an
application's back: pass `KennelConfig(log_events=True)` (or the config key) to get the same
per-session file, or subscribe `kennel.JsonlEventLog(path)` yourself for full control.

The default instructions tell the model to glob, then grep/read, then answer, and include a
one-line overview of the workspace's top-level entries. On the on-device model this is what
makes tool use reliable, especially for non-English prompts.

`Agent(instructions=...)` (or `--instructions` / `kennel.json`'s `agent.instructions`) appends
to those default instructions and is the safe way to add a house rule ("Answer in Japanese").
`Agent(system_prompt=...)` (or `--system-prompt` / `agent.system_prompt`) **replaces** the
default instructions outright; the workspace overview is still appended unless
`include_workspace_overview=False`. Replacing the default instructions removes the
glob-then-read procedure that keeps the on-device model calling tools instead of guessing, so
tool use can become unreliable — only do this if your own instructions cover that ground.

`AppleProvider(deterministic=True)` switches to greedy sampling so a given prompt yields the same trace and answer every run,
which is what you want for evaluations:

```python
from kennel import Agent
from kennel.providers.apple import AppleProvider

agent = Agent(".", provider=AppleProvider(deterministic=True))
```

`provider=` also takes a registered provider's name (`Agent(".", provider="mock")`), the same
names `--provider` offers; leaving it out uses the `provider` config key, defaulting to `apple`.
A name is built with the options under `providers.<name>` in the config. Register your own with
`ProviderSpec`, and `Agent`, the CLI and `kennel doctor` can all use it:

```python
from kennel import ProviderSpec, register_provider

register_provider(ProviderSpec("echo", lambda **options: EchoProvider(**options)))
agent = Agent(".", provider="echo")
```

A provider that needs an API key declares it instead of taking it from the config:
`ProviderSpec("echo", factory, secrets={"api_key": Secret("ECHO_API_KEY")})` makes Kennel read
`ECHO_API_KEY` and pass it as `api_key=`, refuse `providers.echo.api_key` written as a value,
and show in `kennel doctor` whether the variable is set — the same mechanism as the web search
keys below.

### llama-server (a bigger context window, no Apple Intelligence needed)

The on-device model's window is 4096 tokens, which a 20KB file does not fit. The
`llama-server` provider talks to a [llama.cpp](https://github.com/ggml-org/llama.cpp) server
you run yourself, so the window is whatever you started it with. **Kennel never downloads a
model and never starts the server** — that stays your decision:

```bash
brew install llama.cpp
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M --jinja -c 32768 --port 8080 --host 127.0.0.1
# a thinking model: add --reasoning off, or the model thinks before every tool call
kennel ~/meetings --provider llama-server
```

```json
{
  "provider": "llama-server",
  "providers": {
    "llama-server": {
      "base_url": "http://127.0.0.1:8080",
      "model": null,
      "timeout": 300,
      "extra_body": { "chat_template_kwargs": { "enable_thinking": false } }
    }
  }
}
```

- `base_url` defaults to `http://127.0.0.1:8080`. A **loopback** host reports `mode: local`;
  any other host reports `mode: remote` in the CLI header, because inference then leaves this
  machine (see [Local-first contract](#local-first-contract)).
- `model` overrides the name `/v1/models` reports; `timeout` bounds each socket operation;
  `extra_body` is merged into every request body for model-specific knobs. Kennel adds none
  of its own.
- The context window comes from the server's `/props`, and the token counts come from the
  server itself — so with this provider `AgentResult.usage` is filled and `/usage` says
  *reported by the provider* instead of *estimated*.
- Right after `Ctrl-C`, a timeout, a failed turn or `/compact`, `/usage` briefly switches
  back to *estimated*: the provider session that was being counted just got retired, so
  there is nothing left to ask, and the next turn opens a fresh one counted from a summary
  (see `Session.context_usage()`). `/usage` says so: *estimated: no live provider session*.
- A server started with `--api-key` needs the same key in `KENNEL_LLAMA_SERVER_API_KEY`;
  Kennel then sends `Authorization: Bearer <key>` on every request, and sends none when the
  variable is unset. The key is read from the environment only — `providers.llama-server.api_key`
  in a config file is refused, because the agent can read `kennel.json`. To reuse the variable
  llama-server itself reads, name it:
  `{"providers": {"llama-server": {"secrets": {"api_key": "LLAMA_API_KEY"}}}}`. The default is
  deliberately a different name, so a key exported to start a local server is not sent to a
  `base_url` on another host. A refused key fails with HTTP 401 and a message that says whether
  a key was sent; `kennel doctor` shows which variable is read and warns when the key would go
  over plain `http://` to another host (it is still sent: a key is better than none, but use
  `https://` on a network you do not trust).
- Any OpenAI-compatible server (Ollama, LM Studio, mlx-lm) speaks the same protocol, but
  llama-server is the one Kennel is tested against.

### llama-cpp (the same model in this process, no server, no port)

Same llama.cpp, without a server to run: `llama-cpp` loads a GGUF into Kennel's own
process through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Nothing
listens on a port, so nothing but Kennel can reach the model — the strongest form of the
local-first contract, and `mode` is `local` unconditionally. **Kennel still never downloads
a model**: you point it at a `.gguf` you already have.

```bash
uv tool install "git+https://github.com/otiai10/kennel#egg=kennel[llama-cpp]"
# in a checkout: uv sync --extra dev --extra llama-cpp
kennel ~/meetings --provider llama-cpp
```

```json
{
  "provider": "llama-cpp",
  "providers": {
    "llama-cpp": {
      "model_path": "~/models/Qwen3-4B-Q4_K_M.gguf",
      "n_ctx": 16384,
      "n_gpu_layers": -1,
      "chat_template_kwargs": { "enable_thinking": false },
      "llama_kwargs": {}
    }
  }
}
```

- The `[llama-cpp]` extra is **not** installed by default. PyPI ships an sdist only, so
  installing it builds llama.cpp from source: about a minute on Apple Silicon, Xcode
  Command Line Tools required, cmake is fetched automatically. Kennel is verified against
  llama-cpp-python 0.3.35.
- `model_path` is the only required option (`~` is expanded). `n_ctx` is the context window
  and is what the CLI header and `/usage` report; `n_gpu_layers` defaults to `-1` (every
  layer on the GPU); `llama_kwargs` reaches the `Llama` constructor for anything else
  (`n_threads`, `seed`, …).
- **Tool calls are Qwen3-shaped.** Kennel renders the GGUF's own chat template and parses
  the `<tool_call>{json}</tool_call>` the model emits, because llama-cpp-python passes
  `tools` to the template but does not parse what comes back. A GGUF whose template ignores
  `tools` can still chat, but cannot call tools. Qwen3-4B-GGUF is what this is tested
  against.
- **Thinking is off by default**: `chat_template_kwargs` defaults to
  `{"enable_thinking": false}`, because Qwen3 otherwise spends a few hundred tokens in
  `<think>` before every tool call. Pass `{}` to send nothing, or
  `{"enable_thinking": true}` to let it think — either way the scratchpad is stripped and
  never appears in the answer.
- Token counts come from the model's own tokenizer, so `AgentResult.usage` is filled and
  `/usage` says *reported by the provider*. As with `llama-server`, that flips back to
  *estimated* right after `Ctrl-C`, a timeout, a failed turn or `/compact` — the counted
  session was retired, and the next one starts from a summary instead.
- The model is loaded once, on the first turn, and shared by every session; `kennel doctor`
  and `Agent(...)` only check the import, the file and `n_ctx`, so neither waits for a load.
  One model is one KV cache, so turns are run one at a time.
- Ctrl-C stops generation at the next token. While the prompt is still being evaluated
  (about 11 seconds for a 7k-token prompt on an M-series Mac) nothing interrupts it.

Ctrl-C (`Session.interrupt()`), `Session.stream()`, `run(schema=)` and `compact()` all work
the same as on Apple; the tool-calling loop for chat-completions providers is
`kennel/providers/chatloop.py`, shared rather than written per provider.

**Narration guard.** Small models sometimes describe the tool steps they would take
("I'll run grep for that", "which file should I check?") instead of taking them. When a
turn ends with no tool call and the answer looks like that, Kennel re-prompts once inside
the same session to carry the steps out; the CLI shows `↻ carrying out the described
steps`. Set `"nudge_narration": false` under `"agent"` in `kennel.json` to turn it off.

**Structured output.** Pass a JSON schema to get a value instead of prose (Apple guided
generation). Tools keep working: the provider calls them inside the same request, so
`tool_calls` is populated as usual.

```python
result = await agent.run("Extract the decisions from the latest transcript", schema=MEETING_SCHEMA)
result.structured_output          # dict
result.text                       # the same document as JSON
```

`examples/meeting_summary.py` shows the reference workflow: chunk a transcript, extract a
`MeetingSummary` per chunk with `agent.run(schema=...)`, reduce. Owners and due dates that
are not in the transcript stay `None`. `ProviderSession.respond_structured` remains the
provider-level primitive underneath.

### Web search (off by default, no default service)

The `web` tool is off unless you pass `--allow-web` (SDK: `tools=[..., "web"]`), and even then
it only works once you choose a search service; Kennel does not pick one for you. Every search
sends the query off this Mac to that service. Two services are built in:

| Service | What you provide | Where queries go |
| --- | --- | --- |
| `searxng` | `url` of a SearXNG you run, with `json` in `search.formats` of its `settings.yml` | your instance, which relays them to upstream engines |
| `brave` | an API key in `BRAVE_API_KEY` ([Brave Search API](https://brave.com/search/api/); a card is required) | `api.search.brave.com` |

With `brave`, your key and your searches are under Brave's API terms; Kennel keeps no search
results (see below), and any attribution Brave asks of you is yours to give.

```json
{
  "search_provider": "searxng",
  "search_providers": {
    "searxng": { "url": "http://localhost:8888" },
    "brave": { "secrets": { "api_key": "MY_BRAVE_KEY" } }
  }
}
```

API keys are read from environment variables only. A config file may name a different
variable (`secrets.api_key` above), but a key written into `kennel.json` or `settings.json` is
refused: the agent can read the workspace's `kennel.json` with its own `read` tool. Both
services also take `timeout` (seconds, default 10) and `max_response_bytes` (default
1000000); a response larger than that is not read to the end.

The header shows where searches go, on a line of its own; `mode:` stays the model's:

```text
$ kennel ~/meetings --allow-web

Kennel v0.1.0
workspace: /Users/me/meetings
provider: apple
model: Apple SystemLanguageModel
mode: local
permissions: default
tools: glob, grep, read, write (ask), edit (ask), shell (ask), web (ask)
web search: searxng → localhost:8888 → upstream engines (remote)
```

`Session.status()["web"]` (and `/status`) carries the same text. `kennel doctor` reports the
service, its destination and mode, whether each key's variable is set (never its value), and
whether the service is reachable: for SearXNG it sends one probe query, which goes through
your instance to its upstream engines; for Brave it only opens a connection. Events, the
session log and `--output-format` record how many results a search returned
(`{"provider": "searxng", "returned": 5}`), never their text; the results themselves go to
the model only.

A search that could not be carried out fails with `SearchError`, whose message tells the model
whether trying again can help (a timeout) or not (a rejected key, a spent quota, SearXNG's
JSON output disabled, blocked upstream engines), and what to fix. An empty answer means the
service looked and found nothing.

Any other search service plugs in the same way. A provider has an `info` and an async
`search`; register it, and `search_provider` can name it:

```python
from kennel import SearchError, SearchProviderInfo, SearchResult, SearchSpec, Secret, register_search_provider

class MySearch:
    def __init__(self, token: str, base_url: str = "https://search.example"):
        self.token, self.base_url = token, base_url
        # "remote" unless the query never leaves this machine (an offline index)
        self.info = SearchProviderInfo("mysearch", "search.example", "remote")

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        hits = await my_client(self.base_url, self.token, query, limit)   # your HTTP call
        if hits is None:
            raise SearchError("mysearch is down", retryable=True)        # could not look
        return [SearchResult(h.title, h.url, h.snippet) for h in hits]    # [] = nothing found

register_search_provider(SearchSpec(
    "mysearch",
    lambda **options: MySearch(**options),
    secrets={"token": Secret("MYSEARCH_TOKEN")},   # read from the environment, passed as token=
))
```

`base_url` then comes from `search_providers.mysearch` in the config and `token` from
`MYSEARCH_TOKEN`. An application can also skip the registry and pass the tool itself:
`Agent(".", tools=["read", WebSearchTool(MySearch(token))])` (from `kennel.tools.web`); an
instance given that way wins over the config.

### Fetch (off by default, public addresses only)

`fetch` reads one web page by URL and gives the model its text, so a search result can be
read and quoted rather than guessed at from its snippet. It is a separate tool from `web`
with its own permission: `--allow-fetch` enables it and asks before each call (`--allow-web`
does not enable it). In the SDK, list it and give it a rule, like `web`:
`Agent(".", tools=[..., "fetch"], permissions={"fetch": "ask"})`; listed without a rule it is
denied in the default mode.

- **Destinations.** Only `http` and `https`. Every address the host name resolves to must be
  public: a loopback, private, link-local, multicast, reserved or unspecified address (an
  IPv4-mapped IPv6 one included) refuses the request before anything is sent. The connection
  goes to the address that was checked, so the name is not looked up again; `Host`, SNI and
  certificate verification use the name. Nothing in `kennel.json` or on the command line lifts
  this; an application that means to reach its own network builds
  `FetchTool(allow_private_addresses=True)` (from `kennel.tools.fetch`) and passes it in `tools`.
- **Permission per host.** A rule's specifier is matched against the normalised host name only
  (lower case, no trailing dot, IDNA): `"fetch(*.python.org)": "allow"` covers
  `https://docs.python.org/...` but not `https://python.org/` or
  `https://evil.example/?x=docs.python.org`. "Allow for this session" remembers the host too, so
  another host is asked about again.
- **Redirects.** Followed on the same host and port (and from `http` to `https` on the default
  ports), at most 5 times, each hop checked again. A redirect elsewhere, or from `https` to
  `http`, is not followed: the model is told where it points and has to call `fetch` again, which
  goes through the permission check for that host.
- **What comes back.** HTML is reduced to its text (no script or style), other `text/*` and JSON
  as they are; any other type, or a compressed body, is an error. A body over 2 MB stops being
  read, and the whole request, redirects included, has a 15 s deadline (`FetchTool(timeout=...,
  max_bytes=...)`). `Ctrl-C` (`Session.interrupt()`) stops it at any stage; interrupted before the
  request went out, nothing is sent.
- **What is recorded.** The approval prompt shows the whole URL. Events and the session log show
  it without the query and the fragment, and the metadata
  (`{"host", "status", "bytes", "content_type"}`), never the page.

A fetched page is untrusted input to the model: it can contain instructions written to steer the
agent (prompt injection). Keep `fetch` on `ask` for hosts you have not chosen, and do not allow it
together with `shell` or `write` without looking at each call.

## Configuration

Precedence: CLI flags > `Agent(...)` arguments > `./kennel.json` > `~/.config/kennel/settings.json` > defaults.

Config files are JSON so that Kennel can write settings back (for example permission rules
approved from the prompt, planned for a later version). All keys are optional.

```json
{
  "agent": {
    "max_tool_calls": 32,
    "turn_timeout_seconds": 300,
    "nudge_narration": true,
    "instructions": "Answer in Japanese."
  },
  "provider": "apple",
  "providers": {
    "apple": { "deterministic": false },
    "llama-server": { "base_url": "http://127.0.0.1:8080" },
    "llama-cpp": { "model_path": "~/models/Qwen3-4B-Q4_K_M.gguf", "n_ctx": 16384 }
  },
  "search_provider": "searxng",
  "search_providers": { "searxng": { "url": "http://localhost:8888" } },
  "permission_mode": "default",
  "logging": { "events": true },
  "sessions": { "persist": false },
  "permissions": {
    "write": "ask",
    "write(docs/**)": "allow",
    "shell": "ask",
    "shell(git *)": "allow",
    "read(**/.env)": "deny"
  },
  "tools": {
    "read": { "max_lines": 400 },
    "grep": { "max_results": 100 },
    "shell": { "timeout_seconds": 30 }
  }
}
```

## Local-first contract

In the default configuration:

- inference uses the Apple on-device model only;
- a `llama-server` on a loopback address is equally local; pointing `base_url` at another
  host is not, and says so with `mode: remote`;
- `llama-cpp` runs the model inside Kennel's own process, so it opens no port at all and is
  `local` unconditionally; no provider downloads a model;
- local file contents are never sent over the network;
- the `web` tool is disabled, and no search service is chosen: web search needs both
  `--allow-web` and a `search_provider`;
- the `fetch` tool is disabled: it needs `--allow-fetch` (or an explicit rule in the SDK), and
  even then it only reaches public addresses;
- no telemetry is collected and no Kennel backend is involved.

`mode: local` in the CLI header reflects the model provider's declaration. A future provider
that sends data off the device (Private Cloud Compute) must report a different mode so the
UI can show it. A search service declares its own mode (`SearchProviderInfo.mode`, `remote`
for both built-in services), shown on the separate `web search:` line and in
`Session.status()["web"]`.

## Security limitations

Read [SECURITY.md](SECURITY.md) before enabling `shell`. In short: the workspace boundary is
enforced for the file tools (including `..`, absolute paths and symlinks), but it is a
usability and safety control, not an OS sandbox. An approved `shell` command can do
anything your user account can do, so `shell` has its own permission and is never enabled
by `--allow-write`. Dangerous-looking commands add a warning to the prompt; that heuristic
is not a security boundary.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests examples
pytest                                   # unit, provider (MockProvider) and CLI tests; no model needed
KENNEL_APPLE_TESTS=1 pytest -m apple     # Apple integration tests, on a capable Mac only
KENNEL_LLAMA_SERVER=http://127.0.0.1:8080 pytest -m llama   # against a llama-server you started
KENNEL_LLAMA_CPP_MODEL=~/models/qwen.gguf pytest -m llama   # against a GGUF in this process
KENNEL_SEARXNG_URL=http://localhost:8888 pytest -m search   # a SearXNG you run (KENNEL_BRAVE_TESTS=1 + BRAVE_API_KEY: Brave)
```

Repository layout:

```text
src/kennel/
  agent.py session.py runner.py      Agent, Session, tool runner (guardrails, hooks, permissions, events)
  hooks.py                           before_tool / after_tool / before_prompt callbacks
  workspace.py permissions.py rules.py  path resolver, permission manager, glob/rule syntax
  registry.py tools/                  tool interface and built-ins (glob grep read write edit shell web)
  search/ credentials.py              search providers (registry, SearXNG, Brave, bounded HTTP), env-only secrets
  providers/                          provider abstraction, registry (name -> provider), AppleProvider,
                                      LlamaServerProvider, the shared chat loop, MockProvider
  context.py config.py events.py      chunking/compaction, JSON config, event bus
  cli/                                argparse CLI, renderer, JSON output
tests/{unit,providers,e2e,integration}
examples/                             meeting_summary.py, repo_qa.py
spikes/                               Phase 0 SDK experiments (not production code)
```

## Status

v0.1.0: interactive and one-shot CLI with `kennel doctor`, machine-readable output
(`--output-format`, `--json-schema`), permission modes and rule syntax, hooks,
`Session.stream()` / `interrupt()` / `context_usage()`, MockProvider-based test suite,
structured meeting summary example, provider selection (`--provider`) with two llama.cpp
providers for a larger context window (`llama-server` over HTTP, `llama-cpp` in this
process), and the observability work that fills out this release: a per-session JSONL event
log, tool result sizes and remaining context shown without `--verbose`, failed turns kept in
the history with the facts of the failure, and a tool result too big for the window withheld
instead of failing the turn, and opt-in web search through SearXNG or Brave. Not yet:
fetching a web page's contents, forking, listing or naming saved sessions (resuming one is
in: `--continue` / `--resume`), MCP,
subagents, sandboxed shell. Design principles live in
[docs/constitution.md](docs/constitution.md); the comparison with Claude Code that drove the
current roadmap is in `docs/history/`.
