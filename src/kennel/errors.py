"""Kennel error hierarchy.

Every error Kennel raises derives from :class:`KennelError` so consumers can
catch one base class. ``str(error)`` is always a user-facing message; the CLI
prints it without a traceback unless ``--verbose`` is set.
"""

from __future__ import annotations


class KennelError(Exception):
    """Base class for all Kennel errors."""


class ConfigurationError(KennelError):
    """Invalid configuration (file, arguments or environment)."""


class ProviderError(KennelError):
    """The model provider failed."""


class ModelUnavailableError(ProviderError):
    """The model cannot be used on this machine right now."""


class ContextLimitError(ProviderError):
    """The model's context window was exceeded."""


class ToolError(KennelError):
    """Base class for tool errors."""


class ToolArgumentError(ToolError):
    """Tool arguments are missing or invalid."""


class ToolExecutionError(ToolError):
    """The tool ran but failed."""


class ToolOutputLimitError(ToolError):
    """Tool output exceeded a hard limit and could not be delivered."""


class PermissionDeniedError(KennelError):
    """A tool needed a permission that was not granted."""


class WorkspaceError(KennelError):
    """Workspace configuration or path error."""


class WorkspaceEscapeError(WorkspaceError):
    """A path resolves outside the workspace root."""


class SessionError(KennelError):
    """The agent session could not continue."""


class TurnCancelledError(SessionError):
    """The running turn was interrupted through :meth:`kennel.Session.interrupt`.

    The session itself stays usable: the next ``run()`` starts a fresh turn.
    """
