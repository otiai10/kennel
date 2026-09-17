"""``kennel`` command: interactive session or one-shot prompt over a workspace."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from .. import __version__
from ..agent import Agent
from ..errors import ConfigurationError, KennelError, ModelUnavailableError
from ..permissions import PermissionMode
from ..registry import DEFAULT_TOOLS
from ..session import AgentResult, Session
from .renderer import ConsolePrompter, Renderer

PROMPT = "> "


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kennel",
        description="Kennel: a local tool-using agent on Apple Foundation Models.",
        epilog="Examples:\n  kennel .\n  kennel ~/meetings\n  kennel -p 'Read the README and explain this project'",
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
    p.add_argument("--max-tool-calls", type=int, default=None, metavar="N", help="tool call limit per turn (default: 32)")
    p.add_argument("--provider", choices=("apple", "mock"), default="apple", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true", help="show tool result sizes and diagnostics")
    p.add_argument("--trace", action="store_true", help="write every agent event as JSON to stderr")
    p.add_argument("--version", action="version", version=f"kennel {__version__}")
    return p


def _make_provider(name: str):
    if name == "mock":
        from ..providers.mock import MockProvider

        script = os.environ.get("KENNEL_MOCK_SCRIPT", "")
        looks_like_path = script and not script.lstrip().startswith("{") and len(script) < 1024
        if looks_like_path:
            try:
                script = Path(script).read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigurationError(f"KENNEL_MOCK_SCRIPT: cannot read {script!r}: {exc}") from exc
        return MockProvider.from_json(script or "{}")
    from ..providers.apple import AppleProvider

    return AppleProvider()


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

    config = load_config(Path(args.workspace).expanduser()).merged(max_tool_calls=args.max_tool_calls)
    return Agent(
        args.workspace,
        tools=tools,
        permissions=permissions,
        permission_mode=mode,
        provider=_make_provider(args.provider),
        config=config,
        prompter=prompter,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    interactive_permissions = sys.stdin.isatty() and not args.non_interactive
    prompter = ConsolePrompter(color=color) if interactive_permissions else None
    renderer = Renderer(verbose=args.verbose, trace=args.trace, color=color)
    try:
        agent = build_agent(args, prompter)
        agent.check_availability()
        renderer.attach(agent.events)
        if agent.permission_mode is PermissionMode.BYPASS and args.prompt is not None:
            renderer.error(BYPASS_WARNING)  # no header in one-shot mode; warn on stderr
        if args.prompt is not None:
            return run_once(agent, args.prompt, renderer, prompter)
        return run_interactive(agent, renderer, prompter)
    except ModelUnavailableError as exc:
        renderer.error(str(exc))
        return 3
    except ConfigurationError as exc:
        renderer.error(str(exc))
        return 2
    except KennelError as exc:
        renderer.error(str(exc))
        if args.verbose:
            traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # stdout was closed (e.g. `kennel ... | head`): finish quietly.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        return 0


def _run_turn(loop: asyncio.AbstractEventLoop, session: Session, prompt: str, renderer: Renderer, prompter: ConsolePrompter | None) -> AgentResult | None:
    """Run one turn on ``loop``; Ctrl-C cancels the turn and returns None."""
    if prompter is not None:
        prompter.reset()
    task = loop.create_task(session.run(prompt, on_delta=renderer.delta))

    def on_sigint() -> None:
        if prompter is not None:
            prompter.cancel()
        if not task.done():
            task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, on_sigint)
    except (NotImplementedError, RuntimeError):  # pragma: no cover - non-main thread
        pass
    try:
        return loop.run_until_complete(task)
    except asyncio.CancelledError:
        renderer.note("(cancelled)")
        return None
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass


def run_once(agent: Agent, prompt: str, renderer: Renderer, prompter: ConsolePrompter | None) -> int:
    loop = asyncio.new_event_loop()
    session = agent.new_session()
    try:
        result = _run_turn(loop, session, prompt, renderer, prompter)
        renderer.finish_answer()
        if result is None:
            return 130
        if result.stop_reason == "timeout":
            renderer.error("the model did not finish within the turn timeout")
            return 1
        if result.stop_reason == "tool_limit":
            renderer.note("(tool call limit reached; answer may be incomplete)")
        return 0
    finally:
        loop.run_until_complete(session.close())
        loop.close()


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
        f"model: {info.model}",
        f"mode: {info.mode}",
        f"permissions: {agent.permission_mode.value}" + (f" ({rules})" if rules else ""),
        f"tools: {tools}",
    ]
    if agent.permission_mode is PermissionMode.BYPASS:
        lines.append(BYPASS_WARNING)
    lines.append("Type /help for commands, /exit or Ctrl-D to quit.")
    return "\n".join(lines)


HELP = """Commands:
  /help     show this help
  /status   show session, model and tool status
  /clear    forget the conversation
  /exit     leave (also /quit or Ctrl-D)
Ctrl-C cancels the current answer; press it twice at the prompt to exit."""


def run_interactive(agent: Agent, renderer: Renderer, prompter: ConsolePrompter | None) -> int:
    # Deliberately no readline: with libedit, Ctrl-C at the prompt is only acted on at the
    # next Enter and queued input can be dropped. Cooked-mode input() keeps Ctrl-C predictable.
    out = renderer.out
    out.write(_header(agent) + "\n\n")
    out.flush()
    loop = asyncio.new_event_loop()
    session = agent.new_session()
    last_interrupt = 0.0
    try:
        while True:
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
                command = line.split()[0].lower()
                if command in ("/exit", "/quit"):
                    break
                if command == "/help":
                    out.write(HELP + "\n")
                elif command == "/status":
                    for key, value in session.status().items():
                        out.write(f"{key}: {value}\n")
                    out.write(f"tools: {', '.join(agent.tools)}\n")
                    out.write(f"permission_mode: {agent.permission_mode.value}\n")
                    entries = [f"{k}={v.value}" for k, v in sorted(agent.permissions.policy().items()) if k in agent.tools]
                    rules = _rule_list(agent)
                    out.write("permissions: " + ", ".join(entries + ([rules] if rules else [])) + "\n")
                elif command == "/clear":
                    loop.run_until_complete(session.clear())
                    renderer.note("(conversation cleared)")
                else:
                    renderer.note(f"unknown command {command}; try /help")
                out.flush()
                continue
            try:
                result = _run_turn(loop, session, line, renderer, prompter)
            except KennelError as exc:
                renderer.error(str(exc))
                continue
            renderer.finish_answer()
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
