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


class SearchError(ToolExecutionError):
    """A web search could not be carried out (as opposed to finding nothing).

    The model only ever sees ``str(error)`` (``runner._fail``), so the message itself says
    whether retrying can help and what the user should do; ``retryable`` and ``remedy``
    are kept as attributes for SDK callers.
    """

    def __init__(self, message: str, *, retryable: bool, remedy: str | None = None) -> None:
        advice = "Retrying later may help." if retryable else "Repeating this search will not help."
        text = f"{message.rstrip('.')}. {advice}"
        if remedy:
            text += f" {remedy}"
        super().__init__(text)
        self.retryable = retryable
        self.remedy = remedy


class ToolOutputLimitError(ToolError):
    """Tool output exceeded a hard limit and could not be delivered."""


class PermissionDeniedError(KennelError):
    """A tool needed a permission that was not granted."""


class HookError(KennelError):
    """A hook raised, so the work it was guarding failed.

    Hooks are intervention, not observation: a broken policy is an error, not an
    absence of policy (principle 3). ``before_tool`` / ``after_tool`` express that by
    failing the tool call; a ``before_prompt`` hook has no tool call to fail, so the
    turn fails instead and its exception arrives here — as a :class:`KennelError` with
    a message rather than a traceback, which is what lets the CLI stay in its prompt
    loop. The hook's own exception is the ``__cause__``.
    """


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
