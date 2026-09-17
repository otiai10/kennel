from pathlib import Path

import pytest

from kennel.config import KennelConfig, apply_toml, load_config
from kennel.errors import ConfigurationError


def test_defaults():
    cfg = KennelConfig()
    assert cfg.max_tool_calls == 32 and cfg.limits().max_output_bytes == 64 * 1024
    assert cfg.limits().shell_timeout_seconds == 30 and cfg.permissions == {}


def test_precedence_user_then_project_then_overrides(tmp_path: Path):
    user = tmp_path / "user.toml"
    user.write_text('[agent]\nmax_tool_calls = 5\n[permissions]\nwrite = "deny"\nshell = "deny"\n[tools.grep]\nmax_results = 7\n')
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "kennel.toml").write_text('[agent]\nmax_tool_calls = 9\ntools = ["glob", "read"]\ninstructions = "Be brief."\n[permissions]\nwrite = "ask"\n[tools.shell]\ntimeout_seconds = 3\n')
    cfg = load_config(ws, user_config=user)
    assert cfg.max_tool_calls == 9  # project beats user
    assert cfg.grep_max_results == 7  # user value kept
    assert cfg.permissions == {"write": "ask", "shell": "deny"}
    assert cfg.tools == ["glob", "read"] and cfg.instructions == "Be brief."
    assert cfg.shell_timeout_seconds == 3
    assert [Path(s).name for s in cfg.sources] == ["user.toml", "kennel.toml"]
    merged = cfg.merged(max_tool_calls=2, permissions={"write": "allow"}, instructions=None)
    assert merged.max_tool_calls == 2 and merged.permissions == {"write": "allow", "shell": "deny"}
    assert merged.instructions == "Be brief." and cfg.max_tool_calls == 9  # original untouched


def test_missing_files_are_fine(tmp_path: Path):
    cfg = load_config(tmp_path, user_config=tmp_path / "nope.toml")
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
    ],
)
def test_invalid_values(data):
    with pytest.raises(ConfigurationError):
        apply_toml(KennelConfig(), data)


def test_invalid_toml(tmp_path: Path):
    (tmp_path / "kennel.toml").write_text("this is = not [valid")
    with pytest.raises(ConfigurationError, match="invalid TOML"):
        load_config(tmp_path, user_config=None)
    with pytest.raises(ConfigurationError, match="Unknown config key"):
        KennelConfig().merged(bogus=1)
    assert apply_toml(KennelConfig(), {"agent": {"turn_timeout_seconds": None}}).turn_timeout_seconds is None
