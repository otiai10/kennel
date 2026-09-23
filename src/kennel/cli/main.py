"""``kennel`` command: interactive session or one-shot prompt over a workspace."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import __version__
from ..agent import Agent
from ..errors import ConfigurationError, KennelError, ModelUnavailableError, TurnCancelledError
from ..permissions import Decision, PermissionMode
from ..providers.registry import names as provider_names
from ..registry import DEFAULT_TOOLS
from ..session import AgentResult, Session
from .output import FORMATS, DocumentOutput, JsonOutput, MachineOutput
from .renderer import ConsolePrompter, Renderer
from .terminal import discard_typed_ahead

PROMPT = "> "


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kennel",
        description="Kennel: a local tool-using agent on Apple Foundation Models.",
        epilog="Examples:\n  kennel .\n  kennel ~/meetings\n  kennel -p 'Read the README and explain this project'"
        "\n  kennel doctor              # check that this machine can run Kennel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("workspace", nargs="?", default=".", help="workspace directory (default: current directory)")
    p.add_argument("-p", "--prompt", help="run a single prompt and exit")
    p.add_argument(
        "--permission-mode",
        choices=tuple(m.value for m in PermissionMode),
        default=None,
        metavar="MODE",
        help="permission defaults: read-only, default, accept-edits, dont-ask or bypass",
    )
    p.add_argument("--read-only", action="store_true", help="only enable glob, grep and read (= --permission-mode read-only)")
    p.add_argument("--allow-write", action="store_true", help="allow write and edit without asking (= --permission-mode accept-edits)")
    p.add_argument("--allow-shell", action="store_true", help="allow shell without asking (weakens the workspace boundary)")
    p.add_argument("--allow-web", action="store_true", help="enable the web tool (asks; needs a configured search provider)")
    p.add_argument("--non-interactive", action="store_true", help="never prompt; permissions that would ask are denied (= --permission-mode dont-ask)")
    p.add_argument(
        "--instructions",
        metavar="TEXT|@FILE",
        help="append text to the default instructions (or @path/to/file to read it from a file)",
    )
    p.add_argument(
        "--system-prompt",
        metavar="TEXT|@FILE",
        help="replace the default instructions entirely (or @path/to/file); "
        "the model may become less reliable at using tools without them",
    )
    p.add_argument("--max-tool-calls", type=int, default=None, metavar="N", help="tool call limit per turn (default: 32)")
    p.add_argument(
        "--provider",
        choices=provider_names(),
        default=None,
        help="model provider to use (default: the 'provider' config key, otherwise apple)",
    )
    p.add_argument(
        "--output-format",
        choices=FORMATS,
        default="text",
        metavar="FORMAT",
        help="stdout format for -p: text (default), json (one result object), stream-json (one JSON per line)",
    )
    p.add_argument(
        "--json-schema",
        metavar="TEXT|@FILE",
        help="with -p: answer under a JSON schema (guided generation) instead of prose",
    )
    p.add_argument("--verbose", action="store_true", help="add tool timings and diagnostic logging (sizes are shown anyway)")
    p.add_argument("--trace", action="store_true", help="write every agent event as JSON to stderr")
    p.add_argument(
        "--no-log",
        action="store_true",
        help="do not keep this session's event log (= the \"logging\": {\"events\": false} config key); "
        "the log is a local JSON Lines file under KENNEL_STATE_DIR, ~/.local/state/kennel by default",
    )
    p.add_argument("--version", action="version", version=f"kennel {__version__}")
    return p


def _provider_options_from_env() -> dict[str, dict[str, object]] | None:
    """Provider options the environment supplies, in the shape of the ``providers`` config key.

    Only ``KENNEL_MOCK_SCRIPT`` (inline JSON or a path to it) exists today; it becomes the
    ``mock`` provider's ``script`` option, so the registry stays the only place that builds
    providers.
    """
    script = os.environ.get("KENNEL_MOCK_SCRIPT", "")
    if not script:
        return None
    if not script.lstrip().startswith("{") and len(script) < 1024:
        try:
            script = Path(script).read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(f"KENNEL_MOCK_SCRIPT: cannot read {script!r}: {exc}") from exc
    return {"mock": {"script": script}}


def _resolve_text_or_file(value: str | None, workspace: str, flag: str) -> str | None:
    """Return ``value`` as-is, or the contents of the file it points to if it starts with ``@``."""
    if value is None or not value.startswith("@"):
        return value
    path = Path(value[1:]).expanduser()
    if not path.is_absolute():
        path = Path(workspace).expanduser() / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"{flag}: cannot read {path}: {exc}") from exc


def load_schema(value: str | None) -> dict | None:
    """Read a JSON schema given inline or as ``@path``."""
    if value is None:
        return None
    text = value
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(f"--json-schema: cannot read {path}: {exc}") from exc
    try:
        schema = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"--json-schema: invalid JSON ({exc})") from exc
    if not isinstance(schema, dict):
        raise ConfigurationError("--json-schema: the schema must be a JSON object")
    return schema


def exit_code_for(exc: KennelError) -> int:
    """The exit code an error maps to: 3 model unavailable, 2 bad configuration, 1 otherwise."""
    if isinstance(exc, ModelUnavailableError):
        return 3
    if isinstance(exc, ConfigurationError):
        return 2
    return 1


def failure_line(message: str, failure: dict[str, Any] | None) -> str:
    """The error plus what the failed turn actually sent, as one line for the user.

    ``failure`` is :attr:`kennel.Session.last_failure`. Only counted or provider-reported
    numbers go in, and the context figure says when it is an estimate (principle 4).
    """
    if not failure:
        return message
    parts = [message.rstrip(". ") + "."]
    status = failure.get("status")
    if status is not None and str(status) not in message:
        parts.append(f"Provider status {status}.")  # Apple's own message already spells it out
    parts.append(f"This turn added {failure.get('tool_output_bytes', 0):,} bytes of tool output;")
    used = failure.get("context_tokens_before") or 0
    window = failure.get("context_window_tokens")
    qualifier = " (estimated)" if failure.get("estimated") else ""
    if window:
        parts.append(f"context was ~{used:,} / {window:,} tokens{qualifier} before the request.")
    else:
        parts.append(f"context was ~{used:,} tokens{qualifier}, window unknown, before the request.")
    return " ".join(parts)


#: The legacy flags are sugar for a mode, in this precedence order.
MODE_FLAGS: tuple[tuple[str, PermissionMode], ...] = (
    ("read_only", PermissionMode.READ_ONLY),
    ("allow_write", PermissionMode.ACCEPT_EDITS),
    ("non_interactive", PermissionMode.DONT_ASK),
)


def resolve_mode(args: argparse.Namespace) -> PermissionMode:
    """Turn --permission-mode and the legacy flags into one mode."""
    given = [name.replace("_", "-") for name, _ in MODE_FLAGS if getattr(args, name)]
    if args.permission_mode is not None:
        if given:
            raise ConfigurationError(
                "--permission-mode cannot be combined with " + " or ".join(f"--{name}" for name in given)
            )
        return PermissionMode.parse(args.permission_mode)
    if args.read_only and (args.allow_write or args.allow_shell):
        raise ConfigurationError("--read-only cannot be combined with --allow-write or --allow-shell")
    for name, mode in MODE_FLAGS:
        if getattr(args, name):
            return mode
    return PermissionMode.DEFAULT


def build_agent(args: argparse.Namespace, prompter: ConsolePrompter | None) -> Agent:
    mode = resolve_mode(args)
    tools = list(mode.tools() or DEFAULT_TOOLS)
    permissions: dict[str, str] = {}
    if args.allow_shell:
        permissions["shell"] = "allow"
    if args.allow_web:
        tools.append("web")
        if mode is not PermissionMode.BYPASS:
            permissions["web"] = "ask"
    from ..config import load_config

    instructions = _resolve_text_or_file(args.instructions, args.workspace, "--instructions")
    system_prompt = _resolve_text_or_file(args.system_prompt, args.workspace, "--system-prompt")
    config = load_config(Path(args.workspace).expanduser()).merged(
        max_tool_calls=args.max_tool_calls,
        instructions=instructions,
        provider=args.provider,
        providers=_provider_options_from_env(),
    )
    # The CLI keeps a session event log by default, so a failure can be looked at afterwards;
    # the config file can turn it off and --no-log always wins. The SDK default stays opt-in.
    if args.no_log:
        config = config.merged(log_events=False)
    elif config.log_events is None:
        config = config.merged(log_events=True)
    # The provider is left to Agent, which resolves config.provider through the registry.
    return Agent(
        args.workspace,
        tools=tools,
        permissions=permissions,
        permission_mode=mode,
        system_prompt=system_prompt,
        config=config,
        prompter=prompter,
    )


def build_doctor_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kennel doctor",
        description="Check that Kennel can run on this machine and summarize its effective configuration.",
    )
    p.add_argument("workspace", nargs="?", default=".", help="workspace directory to check (default: current directory)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--provider",
        choices=provider_names(),
        default=None,
        help="provider to check (default: the 'provider' config key, otherwise apple)",
    )
    return p


def run_doctor(argv: list[str]) -> int:
    from .doctor import render_json, render_text, run_checks

    args = build_doctor_parser().parse_args(argv)
    checks = run_checks(args.workspace, args.provider)
    ok = all(c.ok for c in checks)
    print(render_json(checks, ok) if args.json else render_text(checks))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "doctor":
        return run_doctor(argv[1:])
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    machine = args.output_format != "text" or args.json_schema is not None
    # In a machine format stdout is JSON only, so anything human (including the
    # permission prompt) moves to stderr.
    human = sys.stderr if machine else sys.stdout
    color = human.isatty() and not os.environ.get("NO_COLOR")
    interactive_permissions = sys.stdin.isatty() and not args.non_interactive
    prompter = ConsolePrompter(out=human, color=color) if interactive_permissions else None
    renderer = Renderer(verbose=args.verbose, trace=args.trace, color=color, quiet=machine)
    # Only established once the flags themselves are valid: until then errors are plain text.
    machine_out: MachineOutput | None = None
    try:
        if machine and args.prompt is None:
            flag = "--json-schema" if args.output_format == "text" else f"--output-format {args.output_format}"
            raise ConfigurationError(f"{flag} needs -p/--prompt")
        schema = load_schema(args.json_schema)
        if args.output_format != "text":
            machine_out = JsonOutput(args.output_format)
        elif schema is not None:
            machine_out = DocumentOutput()
        agent = build_agent(args, prompter)
        agent.check_availability()
        renderer.attach(agent.events)
        if machine_out is not None:
            machine_out.attach(agent.events)
        if agent.permission_mode is PermissionMode.BYPASS and args.prompt is not None:
            renderer.error(BYPASS_WARNING)  # no header in one-shot mode; warn on stderr
        if args.prompt is not None:
            return run_once(agent, args.prompt, renderer, prompter, machine_out, schema)
        return run_interactive(agent, renderer, prompter)
    except KennelError as exc:
        code = exit_code_for(exc)
        if args.verbose and code == 1:  # a bug worth a traceback, not a bad flag or a missing model
            traceback.print_exc()
        return _fail(renderer, machine_out, str(exc), code)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # stdout was closed (e.g. `kennel ... | head`): finish quietly.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        return 0


def _fail(renderer: Renderer, machine_out: MachineOutput | None, message: str, code: int) -> int:
    """Report an error on stderr and, in a machine format, as a result record."""
    renderer.error(message)
    if machine_out is not None:
        machine_out.error(message)
    return code


def _run_turn(
    loop: asyncio.AbstractEventLoop,
    session: Session,
    prompt: str,
    renderer: Renderer,
    prompter: ConsolePrompter | None,
    machine_out: MachineOutput | None = None,
    schema: dict | None = None,
) -> AgentResult | None:
    """Run one turn on ``loop``; Ctrl-C cancels the turn and returns None."""
    if prompter is not None:
        prompter.reset()

    def on_delta(text: str) -> None:
        renderer.delta(text)
        if machine_out is not None:
            machine_out.delta(text)

    task = loop.create_task(session.run(prompt, on_delta=on_delta, schema=schema))

    try:
        loop.add_signal_handler(signal.SIGINT, session.interrupt)
    except (NotImplementedError, RuntimeError):  # pragma: no cover - non-main thread
        pass
    try:
        return loop.run_until_complete(task)
    except (TurnCancelledError, asyncio.CancelledError):
        renderer.note("(cancelled)")
        return None
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass


def run_once(
    agent: Agent,
    prompt: str,
    renderer: Renderer,
    prompter: ConsolePrompter | None,
    machine_out: MachineOutput | None = None,
    schema: dict | None = None,
) -> int:
    loop = asyncio.new_event_loop()
    session = agent.new_session()
    result: AgentResult | None = None
    failed: KennelError | None = None
    failure: dict[str, Any] | None = None
    code = 0
    try:
        try:
            result = _run_turn(loop, session, prompt, renderer, prompter, machine_out, schema)
        except KennelError as exc:
            # Reported here rather than in main() so the failed turn's own facts are still
            # reachable through the session.
            failed, code = exc, exit_code_for(exc)
            if renderer.verbose:
                traceback.print_exc()
        renderer.finish_answer()
        # Shown by default: an on-device window is small enough that "how much is left" is a
        # decision the user makes every turn. note() stays quiet in the machine formats.
        renderer.note(f"(context: {session.context_usage().summary()})")
        failure = session.last_failure
        if failed is not None:
            renderer.error(failure_line(str(failed), failure))
        elif result is None:
            code = 130
        elif result.stop_reason == "timeout":
            renderer.error("the model did not finish within the turn timeout")
            code = 1
        elif result.stop_reason == "tool_limit":
            renderer.note("(tool call limit reached; answer may be incomplete)")
    finally:
        loop.run_until_complete(session.close())  # the closing events precede the result line
        loop.close()
    if machine_out is not None:
        if failed is not None:
            machine_out.error(str(failed), failure=failure)
        elif result is None:
            machine_out.error("the turn was cancelled", stop_reason="cancelled")
        else:
            machine_out.result(result)
    return code


def _usage_lines(session: Session) -> str:
    u = session.context_usage()
    window = "unknown" if u.window_tokens is None else f"{u.window_tokens} tokens"
    used = f"{u.used_tokens} tokens" + (" (estimated)" if u.estimated else " (reported by the provider)")
    ratio = "n/a" if u.window_tokens is None else f"{u.ratio:.0%}"
    return f"window: {window}\nused: {used}\ncontext: {ratio}\nturns: {u.turns}\ncompactions: {u.compactions}\n"


BYPASS_WARNING = "! permission mode bypass: every tool runs without asking, including shell and web"


def _rule_list(agent: Agent) -> str:
    """The ``tool(specifier)=decision`` rules, as shown in the header and /status."""
    return ", ".join(f"{rule.key}={rule.decision.value}" for rule in agent.permissions.rules(specified_only=True))


def _header(agent: Agent) -> str:
    policy = agent.permissions.policy()
    tools = ", ".join(
        name if policy.get(name, "allow") == "allow" else f"{name} ({policy[name].value})"
        for name in agent.tools
    )
    rules = _rule_list(agent)
    info = agent.provider.info
    lines = [
        f"Kennel v{__version__}",
        f"workspace: {agent.workspace.root}",
        f"provider: {info.name}",
        f"model: {info.model}",
        f"mode: {info.mode}",
        f"permissions: {agent.permission_mode.value}" + (f" ({rules})" if rules else ""),
        f"tools: {tools}",
    ]
    web = agent.web_search()
    if web is not None:
        lines.append(f"web search: {web}")
    if agent.permission_mode is PermissionMode.BYPASS:
        lines.append(BYPASS_WARNING)
    lines.append("Type /help for commands, /exit or Ctrl-D to quit.")
    return "\n".join(lines)


_HELP_COMMANDS = (
    ("/help", "show this help"),
    ("/status", "show session, model and tool status"),
    ("/usage", "show how full the model's context window is"),
    ("/clear", "forget the conversation"),
    ("/compact", "summarize the conversation so far to save context"),
    ("/permissions", "show the permission decision for every tool"),
    (
        "/permissions <tool> <mode>",
        "set a tool's decision for this session (allow, ask or deny); does not touch"
        " specifier rules, and deny still wins over them (#17)",
    ),
    ("/exit", "leave (also /quit or Ctrl-D)"),
)
HELP = "Commands:\n" + "\n".join(f"  {name:<28}{desc}" for name, desc in _HELP_COMMANDS) + (
    "\nCtrl-C cancels the current answer; press it twice at the prompt to exit."
)


def _build_commands(
    agent: Agent, session: Session, renderer: Renderer, loop: asyncio.AbstractEventLoop
) -> dict[str, Callable[[list[str]], None]]:
    out = renderer.out

    def cmd_help(_args: list[str]) -> None:
        out.write(HELP + "\n")

    def cmd_status(_args: list[str]) -> None:
        for key, value in session.status().items():
            out.write(f"{key}: {value}\n")
        out.write(f"tools: {', '.join(agent.tools)}\n")
        out.write(f"permission_mode: {agent.permission_mode.value}\n")
        entries = [f"{k}={v.value}" for k, v in sorted(agent.permissions.policy().items()) if k in agent.tools]
        rules = _rule_list(agent)
        out.write("permissions: " + ", ".join(entries + ([rules] if rules else [])) + "\n")

    def cmd_usage(_args: list[str]) -> None:
        out.write(_usage_lines(session))

    def cmd_clear(_args: list[str]) -> None:
        loop.run_until_complete(session.clear())
        renderer.note("(conversation cleared)")

    def cmd_compact(_args: list[str]) -> None:
        turns_before = len(session.history)
        compacted = loop.run_until_complete(session.compact())
        if compacted:
            renderer.note(f"(conversation compacted: {turns_before} turns -> summary)")
        else:
            renderer.note("(nothing to compact)")

    def cmd_permissions(args: list[str]) -> None:
        if not args:
            policy = agent.permissions.policy()
            # Same source as the header/status rule summary (_rule_list): one PermissionManager,
            # one place that knows the precedence (deny > specifier > bare); this only groups its
            # output per tool for display (#22).
            specifier_rules = agent.permissions.rules(specified_only=True)
            out.write(f"mode: {agent.permission_mode.value}\n")
            out.write(f"{'tool':<8}{'decision':<10}{'session-grant':<15}rules\n")
            for name in sorted(agent.tools):
                decision = policy.get(name, Decision.ALLOW)
                scopes = agent.permissions.session_scopes(name)
                grant = "yes" if None in scopes else ",".join(s for s in scopes if s) or "-"
                rules = ", ".join(f"{r.key}={r.decision.value}" for r in specifier_rules if r.tool_name == name)
                out.write(f"{name:<8}{decision.value:<10}{grant:<15}{rules}\n")
            return
        if len(args) != 2:
            renderer.note("usage: /permissions [<tool> allow|ask|deny]")
            return
        tool_name, mode = args
        if tool_name not in agent.tools:
            renderer.note(f"unknown tool {tool_name!r}")
            return
        before = agent.permissions.policy().get(tool_name, Decision.ALLOW)
        try:
            agent.permissions.set_decision(tool_name, mode)
        except ConfigurationError as exc:
            renderer.note(f"error: {exc}")
            return
        renderer.note(f"({tool_name}: {before.value} -> {mode.lower()} for this session)")

    return {
        "/help": cmd_help,
        "/status": cmd_status,
        "/usage": cmd_usage,
        "/clear": cmd_clear,
        "/compact": cmd_compact,
        "/permissions": cmd_permissions,
    }


def run_interactive(agent: Agent, renderer: Renderer, prompter: ConsolePrompter | None) -> int:
    # Deliberately no readline: with libedit, Ctrl-C at the prompt is only acted on at the
    # next Enter and queued input can be dropped. Cooked-mode input() keeps Ctrl-C predictable.
    # For the same reason the prompt owns the tty queue: anything typed while the agent was
    # busy is discarded just before each prompt is printed (#34). Those characters stay on
    # screen because the terminal echoed them at keypress time and we never touch ECHO --
    # a terminal mode we changed is a terminal mode a hard crash can leave broken.
    out = renderer.out
    out.write(_header(agent) + "\n\n")
    out.flush()
    loop = asyncio.new_event_loop()
    session = agent.new_session()
    commands = _build_commands(agent, session, renderer, loop)
    last_interrupt = 0.0
    try:
        while True:
            discard_typed_ahead()  # anything typed while the agent was busy is not this turn
            try:
                line = input(PROMPT)
            except EOFError:
                out.write("\n")
                break
            except KeyboardInterrupt:
                out.write("\n")
                now = time.monotonic()
                if now - last_interrupt < 2.0:
                    break
                last_interrupt = now
                renderer.note("(press Ctrl-C again to exit)")
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                parts = line.split()
                command = parts[0].lower()
                if command in ("/exit", "/quit"):
                    break
                handler = commands.get(command)
                if handler is None:
                    renderer.note(f"unknown command {command}; try /help")
                else:
                    handler(parts[1:])
                out.flush()
                continue
            try:
                result = _run_turn(loop, session, line, renderer, prompter)
            except KennelError as exc:
                renderer.error(failure_line(str(exc), session.last_failure))
                continue
            renderer.finish_answer()
            renderer.note(f"(context: {session.context_usage().summary()})")
            if result is not None and result.stop_reason == "tool_limit":
                renderer.note("(tool call limit reached; answer may be incomplete)")
            elif result is not None and result.stop_reason == "timeout":
                renderer.error("the model did not finish within the turn timeout")
            out.write("\n")
            out.flush()
    finally:
        loop.run_until_complete(session.close())
        loop.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
