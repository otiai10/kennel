"""Kennel: a lightweight local agent runtime and Python SDK for Apple Foundation Models."""

from .agent import DEFAULT_INSTRUCTIONS, Agent
from .config import KennelConfig, load_config
from .credentials import Secret
from .errors import (
    ConfigurationError,
    ContextLimitError,
    HookError,
    KennelError,
    ModelUnavailableError,
    PermissionDeniedError,
    ProviderError,
    SearchError,
    SessionError,
    ToolArgumentError,
    ToolError,
    ToolExecutionError,
    ToolOutputLimitError,
    TurnCancelledError,
    WorkspaceError,
    WorkspaceEscapeError,
)
from .events import (
    Event,
    EventBus,
    EventType,
    JsonlEventLog,
    event_json,
    session_log_path,
    state_dir,
)
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
from .providers.registry import ProviderSpec
from .providers.registry import create as create_provider
from .providers.registry import register as register_provider
from .runner import ToolCallRecord
from .search import SearchProvider, SearchProviderInfo, SearchResult, SearchSpec
from .search import register as register_search_provider
from .session import AgentResult, ContextUsage, Session
from .sessions import FileSessionStore, SessionStore
from .tools.base import Tool, ToolContext, ToolLimits, ToolParameter, ToolResult
from .workspace import Workspace

__version__ = "0.1.0"

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
    "FileSessionStore",
    "HookContext",
    "HookError",
    "JsonlEventLog",
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
    "ProviderSpec",
    "SearchError",
    "SearchProvider",
    "SearchProviderInfo",
    "SearchResult",
    "SearchSpec",
    "Secret",
    "Session",
    "SessionError",
    "SessionStore",
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
    "create_provider",
    "event_json",
    "load_config",
    "register_provider",
    "register_search_provider",
    "session_log_path",
    "state_dir",
]
