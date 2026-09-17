import json
from pathlib import Path

import pytest

from kennel.config import KennelConfig, apply_config, load_config
from kennel.errors import ConfigurationError


def test_defaults():
    cfg = KennelConfig()
    assert cfg.max_tool_calls == 32 and cfg.limits().max_output_bytes == 64 * 1024
    assert cfg.limits().shell_timeout_seconds == 30 and cfg.permissions == {}


def test_precedence_user_then_project_then_overrides(tmp_path: Path):
    user = tmp_path / "settings.json"
    user.write_text(json.dumps({"agent": {"max_tool_calls": 5}, "permissions": {"write": "deny", "shell": "deny"}, "tools": {"grep": {"max_results": 7}}}))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "kennel.json").write_text(json.dumps({"agent": {"max_tool_calls": 9, "tools": ["glob", "read"], "instructions": "Be brief.", "nudge_narration": False}, "permissions": {"write": "ask"}, "tools": {"shell": {"timeout_seconds": 3}}}))
    cfg = load_config(ws, user_config=user)
    assert cfg.max_tool_calls == 9  # project beats user
    assert cfg.grep_max_results == 7  # user value kept
    assert cfg.permissions == {"write": "ask", "shell": "deny"}
    assert cfg.tools == ["glob", "read"] and cfg.instructions == "Be brief."
    assert cfg.shell_timeout_seconds == 3
    assert cfg.nudge_narration is False
    assert [Path(s).name for s in cfg.sources] == ["settings.json", "kennel.json"]
    merged = cfg.merged(max_tool_calls=2, permissions={"write": "allow"}, instructions=None)
    assert merged.max_tool_calls == 2 and merged.permissions == {"write": "allow", "shell": "deny"}
    assert merged.instructions == "Be brief." and cfg.max_tool_calls == 9  # original untouched


def test_system_prompt_precedence(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "kennel.json").write_text(json.dumps({"agent": {"system_prompt": "project prompt"}}))
    cfg = load_config(ws, user_config=None)
    assert cfg.system_prompt == "project prompt"
    merged = cfg.merged(system_prompt="cli prompt")  # CLI flag beats project config
    assert merged.system_prompt == "cli prompt" and cfg.system_prompt == "project prompt"  # original untouched


def test_missing_files_are_fine(tmp_path: Path):
    cfg = load_config(tmp_path, user_config=tmp_path / "nope.json")
    assert cfg == KennelConfig()


@pytest.mark.parametrize(
    "data",
    [
        {"agent": {"max_tool_calls": 0}},
        {"agent": {"max_tool_calls": "many"}},
        {"agent": {"tools": "glob"}},
        {"agent": {"turn_timeout_seconds": -1}},
        {"permissions": {"write": "sometimes"}},
        {"tools": {"read": {"max_lines": True}}},
        {"agent": {"nudge_narration": "yes"}},
        {"agent": {"system_prompt": 123}},
        {"agent": []},
    ],
)
def test_invalid_values(data):
    with pytest.raises(ConfigurationError):
        apply_config(KennelConfig(), data)


def test_invalid_json(tmp_path: Path):
    (tmp_path / "kennel.json").write_text("{not valid json")
    with pytest.raises(ConfigurationError, match="invalid JSON"):
        load_config(tmp_path, user_config=None)
    (tmp_path / "kennel.json").write_text("[1, 2]")
    with pytest.raises(ConfigurationError, match="top level must be a JSON object"):
        load_config(tmp_path, user_config=None)
    with pytest.raises(ConfigurationError, match="Unknown config key"):
        KennelConfig().merged(bogus=1)
    assert apply_config(KennelConfig(), {"agent": {"turn_timeout_seconds": None}}).turn_timeout_seconds is None
