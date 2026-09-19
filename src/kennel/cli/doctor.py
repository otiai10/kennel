"""``kennel doctor``: check that this machine can run Kennel, and how it is configured.

Every check is independent and best-effort: a failure in one (for example an invalid
``kennel.json``) never prevents the others from running. Text output is for humans
(``✓``/``✗`` lines); ``--json`` is for scripts and bug reports.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import (
    PROJECT_CONFIG_NAME,
    USER_CONFIG_PATH,
    KennelConfig,
    load_config,
    read_config_file,
)
from ..diagnostics import Check
from ..errors import ConfigurationError
from ..events import state_dir
from ..providers.registry import DEFAULT_PROVIDER, spec
from ..registry import DEFAULT_TOOLS
from ..workspace import Workspace, WorkspaceError

__all__ = ["Check", "render_json", "render_text", "run_checks"]


def _check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    detail = f"Python {v.major}.{v.minor}.{v.micro}"
    if not ok:
        detail += " (>=3.10 required)"
    return Check("python", ok, detail)


def _check_platform() -> Check:
    mac_version = platform.mac_ver()[0]
    system = f"macOS {mac_version}" if mac_version else platform.system() + " " + platform.release()
    return Check("platform", True, f"{system} ({platform.machine()})")


def _provider_checks(name: str, options: dict[str, Any]) -> list[Check]:
    """Which provider was selected, plus whatever that provider wants checked."""
    try:
        provider_spec = spec(name)
    except ConfigurationError as exc:
        return [Check("provider", False, str(exc), hint="fix 'provider' in the config or pass --provider")]
    checks = [Check("provider", True, f"provider {name}")]
    if provider_spec.doctor_checks is not None:
        try:
            checks.extend(provider_spec.doctor_checks(**options))
        except Exception as exc:  # noqa: BLE001 - one provider's checks must not stop the rest
            checks.append(Check(f"provider {name} checks", False, f"could not run them: {exc}"))
    return checks


def _check_xcode() -> Check:
    try:
        result = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("xcode", True, f"xcode-select unavailable: {exc}")  # informational only
    if result.returncode == 0:
        return Check("xcode", True, f"xcode-select -> {result.stdout.strip()}")
    return Check("xcode", True, "xcode-select: not configured")


def _check_config_file(path: Path, label: str) -> Check:
    if not path.is_file():
        return Check(label, True, f"{label} {path} (not present)")
    try:
        read_config_file(path)
    except ConfigurationError as exc:
        return Check(label, False, str(exc), hint="fix or remove the file")
    return Check(label, True, f"{label} {path} (valid)")


def _check_event_log(cfg: KennelConfig | None) -> Check:
    """Where session event logs go, and whether that directory can be written.

    Read-only: it reports on the nearest directory that already exists rather than creating
    anything, so running ``doctor`` leaves no state behind.
    """
    if cfg is not None and cfg.log_events is False:
        return Check("event log", True, "session event log off ('logging': {'events': false})")
    directory = state_dir() / "sessions"
    existing = directory
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    ok = os.access(existing, os.W_OK)
    detail = f"session event log {directory}" + ("" if ok else f" (not writable: {existing})")
    hint = None if ok else "point KENNEL_STATE_DIR at a writable directory, or run with --no-log"
    return Check("event log", ok, detail, hint=hint)


def _check_workspace(workspace_arg: str) -> tuple[Check, Workspace | None]:
    try:
        ws = Workspace(workspace_arg)
    except WorkspaceError as exc:
        return Check("workspace", False, str(exc)), None
    return Check("workspace", True, f"workspace {ws.root} resolvable"), ws


def _check_effective(cfg: KennelConfig | None, error: str | None, workspace: Workspace | None) -> Check:
    if workspace is None:
        return Check("effective config", False, "skipped: workspace could not be resolved")
    if cfg is None:
        return Check("effective config", False, error or "the config could not be read")
    tools = ",".join(cfg.tools or DEFAULT_TOOLS)
    permissions = ",".join(f"{k}={v}" for k, v in sorted(cfg.permissions.items())) or "(defaults)"
    return Check("effective config", True, f"tools={tools}  permissions={permissions}")


def _load_config(workspace_arg: str, user_config: Path | None) -> tuple[KennelConfig | None, str | None]:
    """The effective config, or the error explaining why it could not be read.

    Read once and shared by the provider selection and the effective-config check, so the
    two can never disagree about what the config says.
    """
    try:
        return load_config(Path(workspace_arg).expanduser(), user_config=user_config), None
    except ConfigurationError as exc:
        return None, str(exc)


def run_checks(
    workspace_arg: str, provider_name: str | None = None, *, user_config: Path | None = USER_CONFIG_PATH
) -> list[Check]:
    """Run every doctor check and return them in display order.

    ``provider_name`` is ``--provider``; ``None`` falls back to the ``provider`` config key
    (and to the default provider when the config cannot be read). ``user_config`` defaults to
    the real ``~/.config/kennel/settings.json`` (matching :func:`kennel.config.load_config`);
    tests override it to avoid depending on the machine's actual home directory.
    """
    cfg, config_error = _load_config(workspace_arg, user_config)
    name = provider_name or (cfg.provider if cfg is not None else DEFAULT_PROVIDER)
    options = cfg.providers.get(name, {}) if cfg is not None else {}
    checks = [_check_python(), _check_platform()]
    checks.extend(_provider_checks(name, options))
    checks.append(_check_xcode())
    if user_config is not None:
        checks.append(_check_config_file(user_config, "user config"))
    checks.append(_check_config_file(Path(workspace_arg).expanduser() / PROJECT_CONFIG_NAME, "project config"))
    checks.append(_check_event_log(cfg))
    workspace_check, workspace = _check_workspace(workspace_arg)
    checks.append(workspace_check)
    checks.append(_check_effective(cfg, config_error, workspace))
    return checks


def render_text(checks: list[Check]) -> str:
    by_name = {c.name: c.detail for c in checks}
    header = f"Kennel {__version__}"
    if "python" in by_name:
        header += "  " + by_name["python"]
    if "platform" in by_name:
        header += "  " + by_name["platform"]
    lines = [header]
    for check in checks:
        if check.name == "effective config":
            lines.append(f"effective: {check.detail}" if check.ok else f"✗ effective config: {check.detail}")
            continue
        mark = "✓" if check.ok else "✗"
        lines.append(f"{mark} {check.detail}")
        if not check.ok and check.hint:
            lines.append(f"  hint: {check.hint}")
    return "\n".join(lines)


def render_json(checks: list[Check], ok: bool) -> str:
    payload: dict[str, Any] = {
        "version": __version__,
        "ok": ok,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail, "hint": c.hint} for c in checks],
    }
    return json.dumps(payload)
