# Kennel

**Kennel is a lightweight local agent runtime and Python SDK for Apple Foundation Models.**

It gives Apple's on-device model a small set of local tools (`glob`, `grep`, `read`, and
permission-gated `write`, `edit`, `shell`), a workspace boundary, a permission layer, an
event stream, and a CLI. Everything runs on your Mac; nothing leaves it by default.

```text
$ kennel ~/meetings

Kennel v0.0.3
workspace: /Users/me/meetings
model: Apple SystemLanguageModel
mode: local
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

Plain pip into an existing environment works too. Pin a tag or branch with `@v0.0.3` at the end of the URL.

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

> **About the `apple-fm-sdk<0.2.1` pin.** The SDK has no wheels; installing it builds a Swift
> bridge with your Xcode. Version 0.2.1 references APIs that only exist in the macOS 27 SDK
> shipped with Xcode 27, so with Xcode 26.x it fails to compile. The upper bound will be
> lifted once a matching Xcode is common.

Check that the model is usable:

```bash
fm available          # Apple's CLI: "System model available"
kennel . -p "hello"   # exit code 3 with a clear message if the model is unavailable
```

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
| `--allow-web` | enable the `web` tool (asks; needs a configured search provider) |
| `--non-interactive` | never prompt; anything that would ask is denied (= `--permission-mode dont-ask`) |
| `--max-tool-calls N` | tool call budget per turn (default 32) |
| `--output-format FORMAT` | with `-p`: `text` (default), `json`, `stream-json` |
| `--json-schema TEXT\|@FILE` | with `-p`: answer under a JSON schema (guided generation) |
| `--verbose` | show tool output sizes and timings |
| `--trace` | write every agent event as JSON lines to stderr |

Interactive commands: `/help`, `/status`, `/usage`, `/clear`, `/exit`. `Ctrl-C` cancels the
current answer via `Session.interrupt()` and returns to the prompt (twice at the prompt exits);
`Ctrl-D` exits.

The on-device model's context window is about 4k tokens, which is the tightest constraint in
practice, so `/usage` shows how full it is. With `--verbose` the same line is printed after
every answer:

```text
> /usage
window: 4096 tokens
used: 504 tokens (estimated)
context: 12%
turns: 1
compactions: 0
```

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
them. `bypass` allows everything including `shell` and prints a warning line in the header.

| Mode | read tools | `write` / `edit` | `shell` | `web` | Tool set |
| --- | --- | --- | --- | --- | --- |
| `read-only` | allow | deny | deny | deny | `glob`, `grep`, `read` only |
| `default` | allow | ask | ask | deny | all |
| `accept-edits` | allow | allow | ask | ask | all |
| `dont-ask` | allow | deny | deny | deny | all |
| `bypass` | allow | allow | allow | allow | all |


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
`session_id`, `duration_ms`, `is_error`, `structured_output`, `compactions`). `usage` is a
`Usage` (`input_tokens`, `output_tokens`, `estimated`); on the on-device model it is estimated,
because the SDK exposes no token counter. `result.to_dict()` gives the same JSON object the CLI
prints with `--output-format json`. Multi-turn conversations use a session:

```python
session = agent.new_session()
await session.run("Summarize transcripts/2026-09-16.txt")
await session.run("Now only the TODOs")
```

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
`session.failed` event has been delivered. `Agent.stream(prompt)` is the one-shot form: it
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
u.turns, u.compactions
print(u.summary())  # "12% of 4096 tokens (504 tokens, estimated)"
```

`used_tokens` counts what is *in the live provider session*: the instructions plus the turns
since it was opened. Compacting the conversation replaces that session with a summarized one,
so the number drops — which is what makes it useful as a budget. A provider that counts tokens
itself can implement `ProviderSession.usage()` and the reported figure is used instead, with
`estimated` False.

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
`bypass` (see the table above); `read-only` also restricts the tool set.

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

Events (`session.started`, `tool.started`, `tool.completed`, `permission.requested`,
`model.delta`, ...) are available through `agent.events.subscribe(callback)`; the CLI
renderer is just one subscriber. Events carry summaries and sizes, never file contents.
`Session.stream()` is the async-iterator view of the same events, for consumers that would
rather `async for` than register a callback.

Events (`session.started`, `session.completed`, `session.failed`, `session.cancelled`,
`model.started`, `model.delta`, `model.completed`, `model.nudged`, `tool.requested`,
`tool.started`, `tool.completed`, `tool.failed`, `permission.requested`,
`permission.denied`, `context.compacted`) are available through
`agent.events.subscribe(callback)`; the CLI renderer is just one subscriber. Events carry
summaries and sizes, never file contents.

The default instructions tell the model to glob, then grep/read, then answer, and include a
one-line overview of the workspace's top-level entries. On the on-device model this is what
makes tool use reliable, especially for non-English prompts. `AppleProvider(deterministic=True)`
switches to greedy sampling so a given prompt yields the same trace and answer every run,
which is what you want for evaluations:

```python
from kennel import Agent
from kennel.providers.apple import AppleProvider

agent = Agent(".", provider=AppleProvider(deterministic=True))
```

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
  "permission_mode": "default",
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
- local file contents are never sent over the network;
- the `web` tool is disabled;
- no telemetry is collected and no Kennel backend is involved.

`mode: local` in the CLI header reflects the provider's declaration. A future provider
that sends data off the device (Private Cloud Compute, a web search provider) must report a
different mode so the UI can show it.

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
```

Repository layout:

```text
src/kennel/
  agent.py session.py runner.py      Agent, Session, tool runner (guardrails, permissions, events)
  workspace.py permissions.py rules.py  path resolver, permission manager, glob/rule syntax
  registry.py tools/                  tool interface and built-ins (glob grep read write edit shell web)
  providers/                          provider abstraction, AppleProvider, MockProvider
  context.py config.py events.py      chunking/compaction, JSON config, event bus
  cli/                                argparse CLI, renderer, JSON output
tests/{unit,providers,e2e,integration}
examples/                             meeting_summary.py, repo_qa.py
spikes/                               Phase 0 SDK experiments (not production code)
```

## Status

v0.0.3. Read-only agent, permission-gated mutation tools, CLI, MockProvider-based test
suite, structured meeting summary example. Not yet: web search provider, persistent
sessions, alternative models, MCP, subagents, sandboxed shell.
