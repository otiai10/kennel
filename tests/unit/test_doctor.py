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
        "web search",  # issue #61: "not configured" when no search_provider is set
        "xcode",
        "user config",
        "project config",
        "event log",
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


def test_event_log_check_reports_the_directory_without_creating_it(meeting_ws: Path, tmp_path: Path, state_dir: Path):
    """doctor is a diagnostic: it says where the log goes and leaves no state behind."""
    checks = run_checks(str(meeting_ws), "mock", user_config=tmp_path / "nope-settings.json")
    log = next(c for c in checks if c.name == "event log")
    assert log.ok and log.detail == f"session event log {state_dir / 'sessions'}"
    assert not state_dir.exists()


def test_event_log_check_says_when_the_config_turned_it_off(meeting_ws: Path, tmp_path: Path):
    (meeting_ws / "kennel.json").write_text(json.dumps({"logging": {"events": False}}))
    checks = run_checks(str(meeting_ws), "mock", user_config=tmp_path / "nope-settings.json")
    log = next(c for c in checks if c.name == "event log")
    assert log.ok and "off" in log.detail


def test_event_log_check_fails_when_the_state_dir_is_not_writable(
    meeting_ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o500)
    monkeypatch.setenv("KENNEL_STATE_DIR", str(blocked / "kennel"))
    try:
        checks = run_checks(str(meeting_ws), "mock", user_config=tmp_path / "nope-settings.json")
        log = next(c for c in checks if c.name == "event log")
        assert not log.ok and "not writable" in log.detail
        assert log.hint is not None and "KENNEL_STATE_DIR" in log.hint
    finally:
        blocked.chmod(0o700)


# -- web search (issue #61, AC-5) ---------------------------------------------------


def _search_checks(ws: Path, tmp_path: Path, config: dict) -> dict[str, Check]:
    (ws / "kennel.json").write_text(json.dumps(config))
    checks = run_checks(str(ws), "mock", user_config=tmp_path / "nope-settings.json")
    return {c.name: c for c in checks if c.name.startswith("web search")}


def test_search_checks_report_name_destination_mode_and_probe(meeting_ws: Path, tmp_path: Path, search_server):
    search_server.reply(200, {"results": [], "unresponsive_engines": []})
    checks = _search_checks(meeting_ws, tmp_path, {"search_provider": "searxng", "search_providers": {"searxng": {"url": search_server.url}}})
    netloc = search_server.url.removeprefix("http://")
    assert checks["web search"].detail == f"web search: searxng → {netloc} → upstream engines (remote)"
    assert checks["web search reachable"].ok
    assert "probe query sent via upstream engines" in checks["web search reachable"].detail
    assert search_server.requests[0][0].endswith("format=json")


def test_searxng_json_disabled_hint(meeting_ws: Path, tmp_path: Path, search_server):
    search_server.reply(403, b"Forbidden")
    checks = _search_checks(meeting_ws, tmp_path, {"search_provider": "searxng", "search_providers": {"searxng": {"url": search_server.url}}})
    reachable = checks["web search reachable"]
    assert not reachable.ok and "search.formats" in reachable.hint and "json" in reachable.hint
    assert "probe query" in reachable.detail


def test_brave_key_presence_without_its_value(meeting_ws: Path, tmp_path: Path, monkeypatch):
    import kennel.search.brave as brave

    probed = []
    monkeypatch.setattr(brave, "probe_connection", lambda endpoint: probed.append(endpoint.netloc))
    monkeypatch.setenv("BRAVE_API_KEY", "doctor-secret-value")
    checks = _search_checks(meeting_ws, tmp_path, {"search_provider": "brave"})
    assert checks["web search"].detail == "web search: brave → api.search.brave.com (remote)"
    assert checks["web search key api_key"].ok and "BRAVE_API_KEY set" in checks["web search key api_key"].detail
    assert "connection only, no query sent" in checks["web search reachable"].detail
    assert probed == ["api.search.brave.com"]
    assert "doctor-secret-value" not in render_text(list(checks.values()))


def test_brave_missing_key_is_reported(meeting_ws: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    checks = _search_checks(meeting_ws, tmp_path, {"search_provider": "brave"})
    key = checks["web search key api_key"]
    assert not key.ok and "BRAVE_API_KEY not set" in key.detail and "environment" in key.hint
    assert checks["web search"].ok and not checks["web search reachable"].ok


def test_no_search_provider_is_reported_and_passes(meeting_ws: Path, tmp_path: Path):
    checks = _search_checks(meeting_ws, tmp_path, {})
    assert checks["web search"].ok and "not configured" in checks["web search"].detail


# -- issue #77: an unreadable config is not replaced by guessed options ----------------------

UNREADABLE_CONFIGS = {
    "invalid json": lambda url: "{not valid json",
    "api_key in the file": lambda url: json.dumps(
        {"provider": "llama-server", "providers": {"llama-server": {"base_url": url, "api_key": "sk-in-the-file"}}}
    ),
}


@pytest.fixture
def no_connections(monkeypatch):
    """Record (and refuse) every HTTP connection doctor tries to open, wherever it points."""
    import http.client

    attempts: list[tuple[str, int]] = []

    def refuse(self):
        attempts.append((self.host, self.port))
        raise OSError("doctor must not connect anywhere in this test")

    monkeypatch.setattr(http.client.HTTPConnection, "connect", refuse)
    return attempts


@pytest.mark.parametrize("provider_flag", [None, "llama-server"])
@pytest.mark.parametrize("config", sorted(UNREADABLE_CONFIGS))
def test_unreadable_config_skips_the_provider_checks(
    meeting_ws: Path, tmp_path: Path, search_server, no_connections, config: str, provider_flag: str | None
):
    """AC-1 / AC-2: no request to the configured server nor to the default base_url."""
    (meeting_ws / "kennel.json").write_text(UNREADABLE_CONFIGS[config](search_server.url))
    checks = run_checks(str(meeting_ws), provider_flag, user_config=tmp_path / "nope-settings.json")
    assert no_connections == [] and search_server.requests == []
    provider_rows = [c for c in checks if c.name.startswith(("provider", "llama-server", "model", "web search"))]
    assert [(c.name, c.ok) for c in provider_rows] == [("provider", False)]
    assert "skipped" in provider_rows[0].detail and "effective config" in provider_rows[0].detail


def test_unreadable_config_keeps_the_independent_checks(meeting_ws: Path, tmp_path: Path, no_connections):
    """AC-3: python / platform / workspace / config files still report; effective config says why."""
    (meeting_ws / "kennel.json").write_text(UNREADABLE_CONFIGS["api_key in the file"]("http://127.0.0.1:8099"))
    checks = run_checks(str(meeting_ws), user_config=tmp_path / "nope-settings.json")
    by_name = {c.name: c for c in checks}
    assert [c.name for c in checks] == [
        "python", "platform", "provider", "xcode", "user config", "project config", "event log", "workspace", "effective config",
    ]
    assert by_name["python"].ok and by_name["platform"].ok and by_name["workspace"].ok and by_name["user config"].ok
    assert by_name["project config"].detail.startswith("project config ")  # reported as before
    assert by_name["effective config"].ok is False
    assert "api_key" in by_name["effective config"].detail
    text = render_text(checks)
    assert "✗ provider skipped: the config could not be read (see effective config)" in text
