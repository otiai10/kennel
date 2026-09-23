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
      "permission_mode": "default",
      "logging": { "events": true },
      "permissions": { "write": "ask", "shell": "ask", "shell(git *)": "allow" },
      "provider": "apple",
      "providers": { "apple": { "deterministic": false } },
      "search_provider": "searxng",
      "search_providers": {
        "searxng": { "url": "http://localhost:8888" },
        "brave": { "secrets": { "api_key": "BRAVE_API_KEY" } }
      },
      "tools": {
        "max_output_bytes": 65536,
        "read": { "max_lines": 400, "max_file_bytes": 2000000 },
        "grep": { "max_results": 100 },
        "glob": { "max_results": 500 },
        "shell": { "timeout_seconds": 30 }
      }
    }

``search_provider`` has no default: web search stays unconfigured until one is chosen, and
the ``web`` tool is only enabled explicitly (``--allow-web``). API keys are never written
here -- ``secrets`` only names the environment variable to read (:mod:`kennel.credentials`).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .permissions import PermissionMode, parse_policy
from .providers.registry import DEFAULT_PROVIDER
from .search import registry as search_registry
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
    #: Keep a session event log under the state directory. ``None`` means "not configured":
    #: the CLI turns it on, an embedding application opts in (see ``kennel.events``).
    log_events: bool | None = None
    permission_mode: str | None = None
    permissions: dict[str, str] = field(default_factory=dict)
    provider: str = DEFAULT_PROVIDER
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    search_provider: str | None = None
    search_providers: dict[str, dict[str, Any]] = field(default_factory=dict)
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
            if key == "providers":
                cfg.providers = _merge_providers(cfg.providers, value)
            elif key == "search_providers":
                cfg.search_providers = _merge_providers(cfg.search_providers, value)
            elif key == "permissions":
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


def _merge_providers(
    base: dict[str, dict[str, Any]], overrides: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Merge provider options per name, with ``overrides`` winning key by key."""
    merged = {name: dict(options) for name, options in base.items()}
    for name, options in overrides.items():
        merged[name] = {**merged.get(name, {}), **options}
    return merged


def _check_search_providers(value: Any, source: str) -> None:
    """``search_providers``: options per name, where a secret may only be a variable name."""
    if not isinstance(value, dict):
        raise ConfigurationError(f"{source}: 'search_providers' must be an object keyed by search provider name")
    for name, options in value.items():
        where = search_registry.where(name)
        if not isinstance(options, dict):
            raise ConfigurationError(f"{source}: {where} must be an object")
        try:
            search_registry.check_options(name, options)
        except ConfigurationError as exc:
            raise ConfigurationError(f"{source}: {exc}") from None


def apply_config(cfg: KennelConfig, data: dict[str, Any], source: str = "<dict>") -> KennelConfig:
    cfg = dataclasses.replace(
        cfg,
        permissions=dict(cfg.permissions),
        providers={name: dict(options) for name, options in cfg.providers.items()},
        search_providers={name: dict(options) for name, options in cfg.search_providers.items()},
        sources=list(cfg.sources),
    )
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
    if "permission_mode" in data:
        cfg.permission_mode = PermissionMode.parse(data["permission_mode"]).value
    logging_section = data.get("logging", {})
    if not isinstance(logging_section, dict):
        raise ConfigurationError(f"{source}: 'logging' must be an object")
    if "events" in logging_section:
        if not isinstance(logging_section["events"], bool):
            raise ConfigurationError(f"{source}: logging.events must be true or false")
        cfg.log_events = logging_section["events"]
    if "system_prompt" in agent:
        if not isinstance(agent["system_prompt"], str):
            raise ConfigurationError(f"{source}: agent.system_prompt must be a string")
        cfg.system_prompt = agent["system_prompt"]
    if "provider" in data:
        if not isinstance(data["provider"], str):
            raise ConfigurationError(f"{source}: 'provider' must be a provider name (a string)")
        cfg.provider = data["provider"]
    if "providers" in data:
        providers = data["providers"]
        if not isinstance(providers, dict):
            raise ConfigurationError(f"{source}: 'providers' must be an object keyed by provider name")
        for name, options in providers.items():
            if not isinstance(options, dict):
                raise ConfigurationError(f"{source}: providers.{name} must be an object")
        cfg.providers = _merge_providers(cfg.providers, providers)
    if "search_provider" in data:
        value = data["search_provider"]
        if value is not None and not isinstance(value, str):
            raise ConfigurationError(f"{source}: 'search_provider' must be a search provider name (a string) or null")
        cfg.search_provider = value
    if "search_providers" in data:
        _check_search_providers(data["search_providers"], source)
        cfg.search_providers = _merge_providers(cfg.search_providers, data["search_providers"])
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
