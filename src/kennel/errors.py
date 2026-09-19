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
    """The model provider failed.

    ``status`` and ``provider_error`` carry what the provider itself reported, and
    only that: a provider that reports no status code leaves ``status`` as ``None``
    rather than having Kennel guess one. ``provider_error`` keeps the provider's
    own wording when Kennel replaces the message with a friendlier one.
    """

    def __init__(self, message: str, *, status: int | None = None, provider_error: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.provider_error = provider_error


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
