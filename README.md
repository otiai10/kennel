# Kennel

**Kennel is a lightweight local agent runtime and Python SDK for Apple Foundation Models.**

It gives Apple's on-device model a small set of local tools (`glob`, `grep`, `read`, and
permission-gated `write`, `edit`, `shell`), a workspace boundary, a permission layer, an
event stream, and a CLI. Everything runs on your Mac; nothing leaves it by default.

```text
$ kennel ~/meetings

Kennel v0.0.1
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

Plain pip into an existing environment works too. Pin a tag or branch with `@v0.0.1` at the end of the URL.

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
| `--read-only` | only `glob`, `grep`, `read` are available |
| `--allow-write` | `write` and `edit` run without asking |
| `--allow-shell` | `shell` runs without asking (see Security) |
| `--allow-web` | enable the `web` tool (asks; needs a configured search provider) |
| `--non-interactive` | never prompt; anything that would ask is denied |
| `--max-tool-calls N` | tool call budget per turn (default 32) |
| `--verbose` | show tool output sizes and timings |
| `--trace` | write every agent event as JSON lines to stderr |

Interactive commands: `/help`, `/status`, `/clear`, `/exit`. `Ctrl-C` cancels the current
answer (twice at the prompt exits); `Ctrl-D` exits.

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
`session_id`). Multi-turn conversations use a session:

```python
session = agent.new_session()
await session.run("Summarize transcripts/2026-09-16.txt")
await session.run("Now only the TODOs")
```

Permissions are per tool (`allow`, `ask`, `deny`); `ask` needs a prompter, otherwise it
means `deny`:

```python
from kennel import Agent, Approval

def prompter(request):            # request.summary, request.details (diff / command), request.warnings
    return Approval.ONCE          # or Approval.SESSION / Approval.DENY

agent = Agent(".", tools=["glob", "grep", "read", "write", "edit", "shell"],
              permissions={"write": "ask", "edit": "ask", "shell": "deny"},
              prompter=prompter)
```

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

Structured output for application use goes through `ProviderSession.respond_structured`
(Apple guided generation). `examples/meeting_summary.py` shows the reference workflow:
chunk a transcript, extract a `MeetingSummary` per chunk, reduce. Owners and due dates
that are not in the transcript stay `None`.

## Configuration

Precedence: CLI flags > `Agent(...)` arguments > `./kennel.toml` > `~/.config/kennel/config.toml` > defaults.

```toml
[agent]
max_tool_calls = 32
turn_timeout_seconds = 300
instructions = "Answer in Japanese."

[permissions]
write = "ask"
edit = "ask"
shell = "deny"

[tools.read]
max_lines = 400

[tools.grep]
max_results = 100

[tools.shell]
timeout_seconds = 30
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
  workspace.py permissions.py         path resolver, permission manager
  registry.py tools/                  tool interface and built-ins (glob grep read write edit shell web)
  providers/                          provider abstraction, AppleProvider, MockProvider
  context.py config.py events.py      chunking/compaction, TOML config, event bus
  cli/                                argparse CLI and renderer
tests/{unit,providers,e2e,integration}
examples/                             meeting_summary.py, repo_qa.py
spikes/                               Phase 0 SDK experiments (not production code)
```

## Status

v0.0.1. Read-only agent, permission-gated mutation tools, CLI, MockProvider-based test
suite, structured meeting summary example. Not yet: web search provider, persistent
sessions, alternative models, MCP, subagents, sandboxed shell.
