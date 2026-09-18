"""Unit tests for the kennel.cli.doctor checks (issues #10, #35)."""

import json
from pathlib import Path

import pytest

from kennel.cli.doctor import Check, render_json, render_text, run_checks
from kennel.providers.registry import _SPECS, ProviderSpec, register


@pytest.fixture
def fake_provider():
    """A third-party provider with its own doctor check (issue #35)."""
    register(
        ProviderSpec(
            "fake",
            lambda **_: None,
            lambda **options: [Check("fake model", True, f"fake ready (options={sorted(options)})")],
        )
    )
    try:
        yield "fake"
    finally:
        _SPECS.pop("fake", None)


def test_mock_provider_checks_pass(meeting_ws: Path, tmp_path: Path):
    checks = run_checks(str(meeting_ws), "mock", user_config=tmp_path / "nope-settings.json")
    assert all(c.ok for c in checks)
    names = [c.name for c in checks]
    assert names == [
        "python",
        "platform",
        "provider",
        "xcode",
        "user config",
        "project config",
        "workspace",
        "effective config",
    ]  # the apple-only checks belong to the apple provider now
    provider = next(c for c in checks if c.name == "provider")
    assert provider.detail == "provider mock"
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


def test_registered_provider_contributes_its_own_checks(meeting_ws: Path, tmp_path: Path, fake_provider: str):
    checks = run_checks(str(meeting_ws), "fake", user_config=tmp_path / "nope-settings.json")
    by_name = {c.name: c for c in checks}
    assert by_name["provider"].detail == "provider fake"
    assert by_name["fake model"].ok and "fake ready" in by_name["fake model"].detail
    assert all(c.ok for c in checks)


def test_provider_comes_from_the_config_when_no_flag_is_given(meeting_ws: Path, tmp_path: Path, fake_provider: str):
    (meeting_ws / "kennel.json").write_text(json.dumps({"provider": "fake", "providers": {"fake": {"port": 8080}}}))
    checks = run_checks(str(meeting_ws), user_config=tmp_path / "nope-settings.json")
    by_name = {c.name: c for c in checks}
    assert by_name["provider"].detail == "provider fake"
    assert "port" in by_name["fake model"].detail  # the configured options reach the checks


def test_unknown_provider_fails_only_the_provider_check(meeting_ws: Path, tmp_path: Path):
    checks = run_checks(str(meeting_ws), "nope", user_config=tmp_path / "nope-settings.json")
    by_name = {c.name: c for c in checks}
    assert by_name["provider"].ok is False
    assert "mock" in by_name["provider"].detail  # the registered names are listed
    assert by_name["python"].ok and by_name["workspace"].ok and by_name["effective config"].ok
