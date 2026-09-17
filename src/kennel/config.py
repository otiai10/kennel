"""Configuration with precedence: CLI flags > Python arguments > project config > user config > defaults.

Config files are JSON (``./kennel.json`` in the workspace, ``~/.config/kennel/settings.json``
for the user). JSON was chosen so that Kennel itself can write settings back (for example
permission rules approved from the prompt) with the standard library alone::

    {
      "agent": {
        "max_tool_calls": 32,
        "turn_timeout_seconds": 300,
        "nudge_narration": true,
        "tools": ["glob", "grep", "read"],
        "instructions": "Answer in Japanese.",
        "system_prompt": "You are a release-notes assistant."
      },
      "permissions": { "write": "ask", "shell": "deny" },
      "tools": {
        "max_output_bytes": 65536,
        "read": { "max_lines": 400, "max_file_bytes": 2000000 },
        "grep": { "max_results": 100 },
        "glob": { "max_results": 500 },
        "shell": { "timeout_seconds": 30 }
      }
    }
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .permissions import parse_policy
from .tools.base import ToolLimits

USER_CONFIG_PATH = Path("~/.config/kennel/settings.json").expanduser()
PROJECT_CONFIG_NAME = "kennel.json"


@dataclass
class KennelConfig:
    max_tool_calls: int = 32
    turn_timeout_seconds: float | None = 300.0
    max_tool_output_bytes: int = 64 * 1024
    max_read_lines: int = 400
    max_file_bytes: int = 2_000_000
    glob_max_results: int = 500
    grep_max_results: int = 100
    shell_timeout_seconds: int = 30
    nudge_narration: bool = True
    permissions: dict[str, str] = field(default_factory=dict)
    tools: list[str] | None = None
    instructions: str | None = None
    system_prompt: str | None = None
    sources: list[str] = field(default_factory=list)

    def limits(self) -> ToolLimits:
        return ToolLimits(
            max_output_bytes=self.max_tool_output_bytes,
            max_read_lines=self.max_read_lines,
            max_file_bytes=self.max_file_bytes,
            glob_max_results=self.glob_max_results,
            grep_max_results=self.grep_max_results,
            shell_timeout_seconds=self.shell_timeout_seconds,
        )

    def merged(self, **overrides: Any) -> KennelConfig:
        """Return a copy with non-None overrides applied (permissions are merged)."""
        cfg = dataclasses.replace(self)
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(cfg, key):
                raise ConfigurationError(f"Unknown config key: {key}")
            if key == "permissions":
                cfg.permissions = {**cfg.permissions, **{k: str(v.value if hasattr(v, "value") else v) for k, v in value.items()}}
            else:
                setattr(cfg, key, value)
        return cfg


_INT_KEYS = {
    ("agent", "max_tool_calls"): "max_tool_calls",
    ("tools", "max_output_bytes"): "max_tool_output_bytes",
    ("tools.read", "max_lines"): "max_read_lines",
    ("tools.read", "max_file_bytes"): "max_file_bytes",
    ("tools.grep", "max_results"): "grep_max_results",
    ("tools.glob", "max_results"): "glob_max_results",
    ("tools.shell", "timeout_seconds"): "shell_timeout_seconds",
}


def apply_config(cfg: KennelConfig, data: dict[str, Any], source: str = "<dict>") -> KennelConfig:
    cfg = dataclasses.replace(cfg, permissions=dict(cfg.permissions), sources=list(cfg.sources))
    agent = data.get("agent", {})
    tools = data.get("tools", {})
    if not isinstance(agent, dict) or not isinstance(tools, dict):
        raise ConfigurationError(f"{source}: 'agent' and 'tools' must be objects")
    sections = {"agent": agent, "tools": tools}
    for name in ("read", "grep", "glob", "shell"):
        sections[f"tools.{name}"] = tools.get(name, {}) if isinstance(tools, dict) else {}
    for (section, key), attr in _INT_KEYS.items():
        if key in sections[section]:
            value = sections[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{source}: {section}.{key} must be a positive integer")
            setattr(cfg, attr, value)
    if "turn_timeout_seconds" in agent:
        value = agent["turn_timeout_seconds"]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
            raise ConfigurationError(f"{source}: agent.turn_timeout_seconds must be a positive number")
        cfg.turn_timeout_seconds = float(value) if value is not None else None
    if "nudge_narration" in agent:
        if not isinstance(agent["nudge_narration"], bool):
            raise ConfigurationError(f"{source}: agent.nudge_narration must be true or false")
        cfg.nudge_narration = agent["nudge_narration"]
    if "tools" in agent:
        value = agent["tools"]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigurationError(f"{source}: agent.tools must be a list of tool names")
        cfg.tools = list(value)
    if "instructions" in agent:
        if not isinstance(agent["instructions"], str):
            raise ConfigurationError(f"{source}: agent.instructions must be a string")
        cfg.instructions = agent["instructions"]
    if "system_prompt" in agent:
        if not isinstance(agent["system_prompt"], str):
            raise ConfigurationError(f"{source}: agent.system_prompt must be a string")
        cfg.system_prompt = agent["system_prompt"]
    perms = data.get("permissions", {})
    if not isinstance(perms, dict):
        raise ConfigurationError(f"{source}: 'permissions' must be an object")
    cfg.permissions.update({k: d.value for k, d in parse_policy(perms).items()})
    cfg.sources.append(source)
    return cfg


def read_config_file(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{path}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"{path}: cannot read config: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path}: top level must be a JSON object")
    return data


def load_config(
    workspace: Path | None = None,
    *,
    user_config: Path | None = USER_CONFIG_PATH,
    project_config_name: str = PROJECT_CONFIG_NAME,
) -> KennelConfig:
    cfg = KennelConfig()
    if user_config is not None and user_config.is_file():
        cfg = apply_config(cfg, read_config_file(user_config), str(user_config))
    if workspace is not None:
        project = Path(workspace) / project_config_name
        if project.is_file():
            cfg = apply_config(cfg, read_config_file(project), str(project))
    return cfg
