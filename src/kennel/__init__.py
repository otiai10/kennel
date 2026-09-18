"""Kennel: a lightweight local agent runtime and Python SDK for Apple Foundation Models."""

from .agent import DEFAULT_INSTRUCTIONS, Agent
from .config import KennelConfig, load_config
from .errors import (
    ConfigurationError,
    ContextLimitError,
    KennelError,
    ModelUnavailableError,
    PermissionDeniedError,
    ProviderError,
    SessionError,
    ToolArgumentError,
    ToolError,
    ToolExecutionError,
    ToolOutputLimitError,
    TurnCancelledError,
    WorkspaceError,
    WorkspaceEscapeError,
)
from .events import Event, EventBus, EventType
from .hooks import HookContext, HookMatcher, Hooks, ToolCallRequest
from .permissions import (
    Allow,
    Approval,
    Decision,
    Deny,
    PermissionKind,
    PermissionManager,
    PermissionMode,
    PermissionOutcome,
    PermissionRequest,
    PermissionRule,
)
from .providers.base import ModelProvider, ProviderInfo, ProviderSession, Usage
from .providers.mock import MockProvider
from .runner import ToolCallRecord
from .session import AgentResult, ContextUsage, Session
from .tools.base import Tool, ToolContext, ToolLimits, ToolParameter, ToolResult
from .workspace import Workspace

__version__ = "0.0.3"

__all__ = [
    "Agent",
    "AgentResult",
    "Allow",
    "Approval",
    "ConfigurationError",
    "ContextLimitError",
    "ContextUsage",
    "DEFAULT_INSTRUCTIONS",
    "Decision",
    "Deny",
    "Event",
    "EventBus",
    "EventType",
    "HookContext",
    "HookMatcher",
    "Hooks",
    "KennelConfig",
    "KennelError",
    "MockProvider",
    "ModelProvider",
    "ModelUnavailableError",
    "PermissionDeniedError",
    "PermissionKind",
    "PermissionManager",
    "PermissionMode",
    "PermissionOutcome",
    "PermissionRequest",
    "PermissionRule",
    "ProviderError",
    "ProviderInfo",
    "ProviderSession",
    "Session",
    "SessionError",
    "Tool",
    "ToolArgumentError",
    "ToolCallRecord",
    "ToolCallRequest",
    "ToolContext",
    "ToolError",
    "ToolExecutionError",
    "ToolLimits",
    "ToolOutputLimitError",
    "ToolParameter",
    "ToolResult",
    "TurnCancelledError",
    "Usage",
    "Workspace",
    "WorkspaceError",
    "WorkspaceEscapeError",
    "load_config",
]
