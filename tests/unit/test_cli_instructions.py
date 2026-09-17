"""CLI wiring for --instructions / --system-prompt (issue #8)."""

import asyncio
from pathlib import Path

import pytest

from kennel.cli.main import build_agent, build_parser
from kennel.errors import ConfigurationError
from kennel.providers.mock import MockProvider


def _agent_instructions(meeting_ws: Path, argv: list[str]) -> str:
    args = build_parser().parse_args([str(meeting_ws), *argv])
    args.provider = "mock"
    agent = build_agent(args, prompter=None)
    agent.provider = MockProvider(["ok"])
    result = asyncio.run(agent.run("x"))
    assert result.text == "ok"
    return agent.provider.sessions[0].instructions


def test_instructions_file_is_appended(meeting_ws: Path):
    (meeting_ws / "extra.md").write_text("Answer only in Japanese.")
    instructions = _agent_instructions(meeting_ws, ["--instructions", "@extra.md"])
    assert "You are Kennel" in instructions
    assert instructions.endswith("Answer only in Japanese.")


def test_instructions_missing_file_raises_configuration_error(meeting_ws: Path):
    args = build_parser().parse_args([str(meeting_ws), "--instructions", "@nope.md"])
    with pytest.raises(ConfigurationError):
        build_agent(args, prompter=None)


def test_system_prompt_replaces_default(meeting_ws: Path):
    instructions = _agent_instructions(meeting_ws, ["--system-prompt", "You are a release bot."])
    assert instructions.startswith("You are a release bot.")
    assert "You are Kennel" not in instructions


def test_system_prompt_cli_overrides_project_config(meeting_ws: Path):
    (meeting_ws / "kennel.json").write_text('{"agent": {"system_prompt": "project prompt"}}')
    instructions = _agent_instructions(meeting_ws, ["--system-prompt", "cli prompt"])
    assert instructions.startswith("cli prompt")
    assert "project prompt" not in instructions
