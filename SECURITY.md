# Security model and limitations

Kennel runs a language model that decides which local tools to call. This document states
what Kennel enforces, what it does not, and how to report a problem.

## What is enforced

**Workspace boundary (file tools).** `glob`, `grep`, `read`, `write` and `edit` resolve every
path through one resolver (`kennel.workspace.Workspace.resolve`). It rejects paths whose
real location is outside the workspace root, including `..` segments, absolute paths,
symlinked files and symlinked parent directories pointing outside, and nonexistent write
destinations whose nearest existing ancestor is outside. Directory listings never follow
symlinks out of the workspace. `.git`, `.venv`, `node_modules` and similar directories are
skipped by default.

**Permissions.** Every tool call resolves to `allow`, `ask` or `deny`. Rules are written per
tool (`shell`) or per tool with a specifier (`shell(git *)`, `write(docs/**)`,
`read(**/.env)`); what a specifier means is decided by the tool (command line for `shell`,
workspace-relative path for the file tools). **`deny` always wins** — a bare `shell: deny`
cannot be reopened by `shell(git *): allow`, and a narrow `shell(git push*): deny` overrides
a broad `shell(git *): allow`. When nothing is denied, a matching specifier rule beats the
bare tool rule and `ask` beats `allow`.

A permission mode supplies the defaults rules are layered on:

| Mode | read tools | `write` / `edit` | `shell` | `web` | Tool set |
| --- | --- | --- | --- | --- | --- |
| `read-only` | allow | deny | deny | deny | `glob`, `grep`, `read` only |
| `default` | allow | ask | ask | deny | all |
| `accept-edits` | allow | allow | ask | ask | all |
| `dont-ask` | allow | deny | deny | deny | all |
| `bypass` | allow | allow | allow | allow | all |

`ask` requires an interactive prompter; without one (piped stdin, `--non-interactive`, an
application that did not provide a prompter) it means `deny`. Approvals are per call or per
session and are never persisted.

`--permission-mode bypass` turns off every prompt, `shell` included. It exists for
non-interactive runs where the caller has accepted that risk; the CLI prints a warning line
in the session header (and on stderr in `-p` mode) whenever it is active. Do not use it on a
workspace you do not fully control.

**Application hooks.** An embedding application can register `before_tool` hooks that run
after argument validation and **before** the permission check, so an application policy can
deny calls the permission policy would allow (`kennel.Hooks`). Through their return value,
hooks only tighten: an `Allow` still goes through the normal permission check, and the only
thing it can skip is the interactive prompt (`Allow(remember="session")` — the same grant
the user could give from the prompt). Hook-rewritten arguments are re-validated and still
resolved through the workspace boundary. A hook that raises fails the tool call rather than
being ignored, so a crashing policy cannot silently disable itself. `before_tool` and
`after_tool` fail the tool call; a `before_prompt` hook has no tool call to fail, so it fails
the turn instead — `Session.run()` raises `HookError` (a `KennelError`) after emitting
`session.failed`.

Hooks are application code running in the same process, not a confined extension point.
`HookContext` hands them the live `PermissionManager` and the `Tool`, so a hook *can* widen
the policy (`ctx.permissions.set_decision(...)`) or call `tool.execute()` directly and
bypass this file's guarantees. That is the same trust level as the application's own code;
do not load hooks you would not paste into your own `main()`.

**Bounded execution.** Tool output is capped (64 KiB by default), reads are line-range
bounded, `shell` has a timeout and runs without stdin, a web search has a deadline for the
whole request and stops reading a response past `max_response_bytes` (1 MB by default), and
each turn has a tool call budget plus repeated-call detection.

**Local by default.** No network access is made by Kennel core. The `web` tool is disabled
unless explicitly enabled (`--allow-web`, which asks before every search) and given a search
provider; no search provider is chosen by default. Search results reach the model only:
events and the session log record how many results came back, not their text.

**Keys stay in the environment.** A search provider's API key is read from an environment
variable only; a key written into `kennel.json` or the user settings file is refused, because
the agent can read the workspace's `kennel.json` with its own `read` tool and a prompt
injection could then send the key out inside a query. Key values never appear in errors,
events, `kennel doctor` or `/status`, and they are not passed to `shell` (whose environment
is an allowlist).

## What is not enforced

- **Kennel is not a sandbox.** The permission layer is a usability and safety control that
  runs in the same process as the model callbacks. There is no macOS App Sandbox, seatbelt
  profile, or filesystem allowlist at the OS level.
- **`shell` bypasses the workspace boundary.** An approved command runs as your user with
  `/bin/sh -c` and can read or modify anything your account can. That is why `shell` has its
  own permission, why `--allow-write` does not enable it, and why `--read-only` removes it
  entirely. The dangerous-command heuristics (`rm -rf`, `sudo`, `git reset --hard`, paths
  outside the workspace, ...) only add a warning to the approval prompt.
- **The environment passed to `shell` is an allowlist** (`PATH`, `HOME`, `LANG`, `TERM`,
  ...) plus anything the application adds explicitly, but a command can still read files
  containing secrets.
- **Model output is untrusted.** Kennel validates and coerces tool arguments and enforces
  the boundary regardless of what the model claims, but it cannot verify that a model's
  natural-language answer is correct.
- **Guardrails are Apple's.** Content safety comes from the on-device model's own guardrails;
  Kennel does not add content filtering.

## Reporting

Open an issue at https://github.com/otiai10/kennel/issues describing the class of problem.
For anything that could expose data outside a workspace, please avoid posting a working
exploit in public and describe the affected code path instead.
