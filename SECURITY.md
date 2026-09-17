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

**Permissions.** Each tool has a policy: `allow`, `ask` or `deny`. Defaults are `allow` for
read-only tools, `ask` for `write`, `edit` and `shell`, `deny` for `web`. `ask` requires an
interactive prompter; without one (piped stdin, `--non-interactive`, an application that
did not provide a prompter) it means `deny`. Approvals are per call or per session and are
never persisted.

**Application hooks.** An embedding application can register `before_tool` hooks that run
after argument validation and **before** the permission check, so an application policy can
deny calls the permission policy would allow (`kennel.Hooks`). Hooks tighten, they do not
loosen: `Allow` from a hook still goes through the normal permission check, and the only
thing it can skip is the interactive prompt (`Allow(remember="session")`, the same grant the
user could give). A hook that raises fails the tool call rather than being ignored, so a
crashing policy cannot silently disable itself. Hook-rewritten arguments are re-validated
and still resolved through the workspace boundary.

**Bounded execution.** Tool output is capped (64 KiB by default), reads are line-range
bounded, `shell` has a timeout and runs without stdin, and each turn has a tool call
budget plus repeated-call detection.

**Local by default.** No network access is made by Kennel core. The `web` tool is disabled
unless explicitly enabled and given a search provider.

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
