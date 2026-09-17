"""Agent: the public entry point that wires workspace, tools, permissions, provider and events."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from .config import KennelConfig, load_config
from .events import EventBus
from .permissions import Decision, PermissionManager, Prompter
from .providers.base import ModelProvider
from .registry import DEFAULT_TOOLS, ToolRegistry, builtin_registry
from .session import AgentResult, Session
from .tools.base import Tool, ToolContext
from .workspace import Workspace, workspace_overview

DEFAULT_INSTRUCTIONS = """You are Kennel, a local assistant with tools for the user's files.
Procedure for every request about files: 1) call glob to find candidate files, 2) call grep or read on the relevant ones, 3) only then answer.
Do not explain or list the steps you are going to take; take them by calling the tools.
For requests that do not involve files, just answer directly without tools.
Never claim to have read a file unless you read it. Paths are relative to the workspace.
Respect permissions: if an action is denied, do not retry it; explain what you would have done.
Answer concisely, grounded in the tool results, in the language of the user's request."""


def _default_provider() -> ModelProvider:
    from .providers.apple import AppleProvider

    return AppleProvider()


class Agent:
    """Configure once, then :meth:`run` prompts or open :meth:`new_session` for multi-turn use.

    Example::

        agent = Agent(workspace="~/meetings", tools=["glob", "grep", "read"])
        result = await agent.run("Find the latest transcript and summarize it")
        print(result.text)
    """

    def __init__(
        self,
        workspace: str | os.PathLike[str] = ".",
        *,
        tools: Iterable[str | Tool] | None = None,
        permissions: Mapping[str, Decision | str] | None = None,
        provider: ModelProvider | None = None,
        instructions: str | None = None,
        config: KennelConfig | None = None,
        prompter: Prompter | None = None,
        events: EventBus | None = None,
        environment: Mapping[str, str] | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.workspace = Workspace(workspace)
        self.config = config if config is not None else load_config(self.workspace.root)
        reg = registry or builtin_registry()
        resolved = reg.resolve(tools, self.config.tools or DEFAULT_TOOLS)
        self.tools: dict[str, Tool] = {t.name: t for t in resolved}
        policy: dict[str, Decision | str] = dict(self.config.permissions)
        policy.update(permissions or {})
        self.permissions = PermissionManager(policy, prompter=prompter)
        self.provider: ModelProvider = provider if provider is not None else _default_provider()
        self.events = events if events is not None else EventBus()
        self.environment: dict[str, str] = dict(environment or {})
        self.instructions = self._build_instructions(instructions)

    def _build_instructions(self, extra: str | None) -> str:
        parts = [DEFAULT_INSTRUCTIONS, f"Workspace root: {self.workspace.root}\n{workspace_overview(self.workspace)}"]
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

    def new_session(self) -> Session:
        return Session(self)

    async def run(self, prompt: str, **kwargs) -> AgentResult:
        """Run ``prompt`` in a fresh session and return the result."""
        session = self.new_session()
        try:
            return await session.run(prompt, **kwargs)
        finally:
            await session.close()

    @property
    def root(self) -> Path:
        return self.workspace.root
