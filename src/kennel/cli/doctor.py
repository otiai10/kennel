"""``kennel doctor``: check that this machine can run Kennel, and how it is configured.

Every check is independent and best-effort: a failure in one (for example an invalid
``kennel.json``) never prevents the others from running. Text output is for humans
(``✓``/``✗`` lines); ``--json`` is for scripts and bug reports.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import PROJECT_CONFIG_NAME, USER_CONFIG_PATH, load_config, read_config_file
from ..errors import ConfigurationError, ModelUnavailableError
from ..registry import DEFAULT_TOOLS
from ..workspace import Workspace, WorkspaceError


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    hint: str | None = None


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


def _check_apple_sdk() -> Check:
    try:
        import apple_fm_sdk as fm
    except ImportError as exc:
        return Check(
            "apple_fm_sdk",
            False,
            f"apple_fm_sdk is not installed ({exc})",
            hint="pip install 'apple-fm-sdk==0.2.0' (requires macOS 26+, Apple Silicon, Xcode)",
        )
    version = getattr(fm, "__version__", "unknown")
    return Check("apple_fm_sdk", True, f"apple_fm_sdk {version} importable")


def _check_apple_availability() -> Check:
    try:
        from ..providers.apple import AppleProvider

        AppleProvider().check_availability()
    except ModelUnavailableError as exc:
        return Check(
            "model availability",
            False,
            str(exc),
            hint="see the detail above for what to change in System Settings",
        )
    except Exception as exc:  # noqa: BLE001 - apple_fm_sdk missing/broken; already reported by the sdk check
        return Check("model availability", False, f"could not check availability: {exc}")
    return Check("model availability", True, "Apple Foundation Models available (SystemLanguageModel)")


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


def _check_workspace(workspace_arg: str) -> tuple[Check, Workspace | None]:
    try:
        ws = Workspace(workspace_arg)
    except WorkspaceError as exc:
        return Check("workspace", False, str(exc)), None
    return Check("workspace", True, f"workspace {ws.root} resolvable"), ws


def _check_effective(workspace: Workspace | None, user_config: Path | None) -> Check:
    if workspace is None:
        return Check("effective config", False, "skipped: workspace could not be resolved")
    try:
        cfg = load_config(workspace.root, user_config=user_config)
    except ConfigurationError as exc:
        return Check("effective config", False, str(exc))
    tools = ",".join(cfg.tools or DEFAULT_TOOLS)
    permissions = ",".join(f"{k}={v}" for k, v in sorted(cfg.permissions.items())) or "(defaults)"
    return Check("effective config", True, f"tools={tools}  permissions={permissions}")


def run_checks(workspace_arg: str, provider_name: str, *, user_config: Path | None = USER_CONFIG_PATH) -> list[Check]:
    """Run every doctor check and return them in display order.

    ``user_config`` defaults to the real ``~/.config/kennel/settings.json`` (matching
    :func:`kennel.config.load_config`); tests override it to avoid depending on the
    machine's actual home directory.
    """
    checks = [_check_python(), _check_platform()]
    if provider_name == "apple":
        checks.append(_check_apple_sdk())
        checks.append(_check_apple_availability())
    else:
        checks.append(Check("apple_fm_sdk", True, f"skipped (--provider {provider_name})"))
        checks.append(Check("model availability", True, f"skipped (--provider {provider_name})"))
    checks.append(_check_xcode())
    if user_config is not None:
        checks.append(_check_config_file(user_config, "user config"))
    checks.append(_check_config_file(Path(workspace_arg).expanduser() / PROJECT_CONFIG_NAME, "project config"))
    workspace_check, workspace = _check_workspace(workspace_arg)
    checks.append(workspace_check)
    checks.append(_check_effective(workspace, user_config))
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
