"""Unit tests for the kennel.cli.doctor checks (issue #10)."""

import json
from pathlib import Path

from kennel.cli.doctor import render_json, render_text, run_checks


def test_mock_provider_checks_pass(meeting_ws: Path, tmp_path: Path):
    checks = run_checks(str(meeting_ws), "mock", user_config=tmp_path / "nope-settings.json")
    assert all(c.ok for c in checks)
    names = [c.name for c in checks]
    assert names == [
        "python",
        "platform",
        "apple_fm_sdk",
        "model availability",
        "xcode",
        "user config",
        "project config",
        "workspace",
        "effective config",
    ]
    apple_sdk = next(c for c in checks if c.name == "apple_fm_sdk")
    assert "skipped" in apple_sdk.detail
    effective = next(c for c in checks if c.name == "effective config")
    assert "tools=" in effective.detail and "permissions=" in effective.detail


def test_invalid_project_config_fails_that_check_only(meeting_ws: Path, tmp_path: Path):
    (meeting_ws / "kennel.json").write_text("{not valid json")
    checks = run_checks(str(meeting_ws), "mock", user_config=tmp_path / "nope-settings.json")
    by_name = {c.name: c for c in checks}
    assert by_name["project config"].ok is False
    assert by_name["project config"].hint is not None
    assert by_name["python"].ok and by_name["platform"].ok and by_name["workspace"].ok
    assert by_name["effective config"].ok is False  # load_config also fails on the same file


def test_missing_workspace_fails_workspace_and_effective(tmp_path: Path):
    missing = tmp_path / "nope"
    checks = run_checks(str(missing), "mock", user_config=tmp_path / "nope-settings.json")
    by_name = {c.name: c for c in checks}
    assert by_name["workspace"].ok is False
    assert by_name["effective config"].ok is False
    assert by_name["python"].ok  # other checks still ran


def test_render_json_is_parseable(meeting_ws: Path, tmp_path: Path):
    checks = run_checks(str(meeting_ws), "mock", user_config=tmp_path / "nope-settings.json")
    ok = all(c.ok for c in checks)
    payload = json.loads(render_json(checks, ok))
    assert payload["ok"] is ok
    assert {c["name"] for c in payload["checks"]} == {c.name for c in checks}
    assert "version" in payload


def test_render_text_marks_ok_and_failing_checks(meeting_ws: Path, tmp_path: Path):
    checks = run_checks(str(meeting_ws), "mock", user_config=tmp_path / "nope-settings.json")
    text = render_text(checks)
    assert text.startswith("Kennel ")
    assert "✓" in text and "✗" not in text
    assert "effective: tools=" in text
