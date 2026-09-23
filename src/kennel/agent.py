"""Agent: the public entry point that wires workspace, tools, permissions, provider and events."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .config import KennelConfig, load_config
from .events import Event, EventBus
from .hooks import Hooks
from .permissions import DEFAULT_MODE, Decision, PermissionManager, PermissionMode, Prompter
from .providers.base import ModelProvider
from .providers.registry import create as create_provider
from .registry import DEFAULT_TOOLS, ToolRegistry, builtin_registry
from .session import AgentResult, Session
from .tools.base import Tool, ToolContext
from .workspace import Workspace, workspace_overview

if TYPE_CHECKING:
    from .sessions import SessionStore

DEFAULT_INSTRUCTIONS = """You are Kennel, a local assistant with tools for the user's files.
Procedure for every request about files: 1) call glob to find candidate files, 2) read the relevant ones, 3) only then answer.
Use grep only to locate an exact word the user gave; for summaries, decisions, tasks or questions about content, read the file. If grep finds nothing, read the file before concluding anything.
Do not explain or list the steps you are going to take; take them by calling the tools.
For requests that do not involve files, just answer directly without tools.
Never claim to have read a file unless you read it. Paths are relative to the workspace.
Respect permissions: if an action is denied, do not retry it; explain what you would have done.
Answer concisely, grounded in the tool results, in the language of the user's request.
End with one short sentence proposing the next step you can take with your tools (for example drafting a reply, saving a summary to a file, or checking another file), when there is one."""


class Agent:
    """Configure once, then :meth:`run` prompts or open :meth:`new_session` for multi-turn use.

    Example::

        agent = Agent(workspace="~/meetings", tools=["glob", "grep", "read"])
        result = await agent.run("Find the latest transcript and summarize it")
        print(result.text)

    ``provider`` takes a :class:`~kennel.ModelProvider` or a registered provider name
    (``Agent(provider="mock")``); ``None`` uses the ``provider`` config key, defaulting to
    ``apple``. A name is built with the options under ``providers.<name>`` in the config.

    ``session_store`` saves every session's turns so a later :meth:`new_session` can pick
    the conversation up again (``new_session(session_id=...)``); without one nothing is saved
    or read back. See :class:`~kennel.FileSessionStore`.
    """

    def __init__(
        self,
        workspace: str | os.PathLike[str] = ".",
        *,
        tools: Iterable[str | Tool] | None = None,
        permissions: Mapping[str, Decision | str] | None = None,
        permission_mode: PermissionMode | str | None = None,
        hooks: Hooks | None = None,
        provider: ModelProvider | str | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        include_workspace_overview: bool = True,
        config: KennelConfig | None = None,
        prompter: Prompter | None = None,
        events: EventBus | None = None,
        environment: Mapping[str, str] | None = None,
        registry: ToolRegistry | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.workspace = Workspace(workspace)
        self.config = config if config is not None else load_config(self.workspace.root)
        mode_spec = permission_mode if permission_mode is not None else self.config.permission_mode
        self.permission_mode = PermissionMode.parse(mode_spec) if mode_spec is not None else DEFAULT_MODE
        reg = registry or builtin_registry()
        default_tools = self.config.tools or self.permission_mode.tools() or DEFAULT_TOOLS
        resolved = reg.resolve(tools, default_tools, self.config)
        self.tools: dict[str, Tool] = {t.name: t for t in resolved}
        policy: dict[str, Decision | str] = dict(self.config.permissions)
        policy.update(permissions or {})
        self.permissions = PermissionManager(policy, prompter=prompter, mode=self.permission_mode)
        self.hooks = hooks if hooks is not None else Hooks()
        self.provider: ModelProvider = self._resolve_provider(provider)
        self.events = events if events is not None else EventBus()
        self.environment: dict[str, str] = dict(environment or {})
        self.session_store = session_store
        # Precedence: Agent() argument > project config > user config. The CLI passes its
        # --system-prompt as this argument, which is what keeps "CLI flags first" true.
        self.system_prompt: str | None = system_prompt if system_prompt is not None else self.config.system_prompt
        self.include_workspace_overview = include_workspace_overview
        self.instructions = self._build_instructions(instructions)

    def _resolve_provider(self, provider: ModelProvider | str | None) -> ModelProvider:
        """An instance wins over any configuration; a name is resolved by the registry."""
        if isinstance(provider, ModelProvider):
            return provider
        name = provider if provider is not None else self.config.provider
        return create_provider(name, **self.config.providers.get(name, {}))

    def web_search(self) -> str | None:
        """Where the ``web`` tool sends searches (``"searxng → host → upstream engines (remote)"``).

        ``None`` when ``web`` is not enabled or no search provider is set up. This is the
        search service's own mode; ``provider.info.mode`` stays the model's.
        """
        from .tools.web import WebSearchTool

        tool = self.tools.get("web")
        return tool.describe() if isinstance(tool, WebSearchTool) else None

    def _build_instructions(self, extra: str | None) -> str:
        base = self.system_prompt if self.system_prompt is not None else DEFAULT_INSTRUCTIONS
        parts = [base]
        if self.include_workspace_overview:
            parts.append(f"Workspace root: {self.workspace.root}\n{workspace_overview(self.workspace)}")
        if self.config.instructions:
            parts.append(self.config.instructions.strip())
        if extra:
            parts.append(extra.strip())
        return "\n\n".join(parts)

    def tool_context(self) -> ToolContext:
        return ToolContext(
            workspace=self.workspace,
            permission_manager=self.permissions,
            environment=self.environment,
            limits=self.config.limits(),
        )

    def check_availability(self) -> None:
        self.provider.check_availability()

    def new_session(self, session_id: str | None = None) -> Session:
        """Open a session; with ``session_id`` and a :attr:`session_store`, resume that conversation.

        The restored session reports ``resumed`` True when saved turns were found (an id with
        nothing saved starts empty under that id). With a store, the id must be 1-64 letters,
        digits, ``-`` or ``_``, and an id another open session holds is refused; both raise
        :class:`~kennel.errors.ConfigurationError`.
        """
        return Session(self, session_id=session_id)

    async def run(self, prompt: str, **kwargs) -> AgentResult:
        """Run ``prompt`` in a fresh session and return the result.

        Keyword arguments go to :meth:`Session.run` — ``on_delta`` to stream the answer,
        ``schema`` for a JSON-schema-shaped :attr:`AgentResult.structured_output`.
        """
        session = self.new_session()
        try:
            return await session.run(prompt, **kwargs)
        finally:
            await session.close()

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Run ``prompt`` in a fresh session and yield its events as they happen.

        The last event is ``session.completed``; its ``data["text"]`` is the same
        answer :meth:`run` returns. Use :meth:`new_session` when you need the
        ``AgentResult`` itself or more than one turn.
        """
        session = self.new_session()
        try:
            async for event in session.stream(prompt):
                yield event
        finally:
            await session.close()

    @property
    def root(self) -> Path:
        return self.workspace.root
