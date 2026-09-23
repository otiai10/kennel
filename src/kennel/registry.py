"""Registry of tool factories and resolution of tool specs into instances."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from .errors import ConfigurationError
from .permissions import READ_ONLY_TOOL_NAMES
from .tools.base import Tool

if TYPE_CHECKING:
    from .config import KennelConfig

ToolFactory = Callable[..., Tool]


class ToolRegistry:
    """Names to tool factories: the only place a tool name becomes a tool instance.

    A factory registered with ``configured=True`` is called with the effective
    :class:`~kennel.config.KennelConfig` (``web`` reads its search provider from it); the
    others are called with no arguments.
    """

    def __init__(self) -> None:
        self._factories: dict[str, ToolFactory] = {}
        self._configured: set[str] = set()

    def register(self, name: str, factory: ToolFactory, *, configured: bool = False) -> None:
        self._factories[name] = factory
        if configured:
            self._configured.add(name)
        else:
            self._configured.discard(name)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def create(self, name: str, config: KennelConfig | None = None) -> Tool:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown tool {name!r}; available: {', '.join(self.names())}"
            ) from exc
        if name in self._configured:
            if config is None:
                from .config import KennelConfig

                config = KennelConfig()
            return factory(config)
        return factory()

    def resolve(
        self,
        specs: Iterable[str | Tool] | None,
        default: Iterable[str],
        config: KennelConfig | None = None,
    ) -> list[Tool]:
        """Turn names and/or instances into a list of unique tool instances.

        An instance is used as given, so it wins over anything the config would build.
        """
        items = list(default) if specs is None else list(specs)
        tools: list[Tool] = []
        seen: set[str] = set()
        for spec in items:
            tool = self.create(spec, config) if isinstance(spec, str) else spec
            if not isinstance(tool, Tool):
                raise ConfigurationError(f"Not a Tool: {spec!r}")
            if tool.name in seen:
                raise ConfigurationError(f"Duplicate tool name: {tool.name!r}")
            seen.add(tool.name)
            tools.append(tool)
        return tools


def _web_tool(config: KennelConfig) -> Tool:
    """``web`` with the search provider the config selects; unconfigured without one.

    A missing API key does not stop the agent from starting: the tool then explains which
    variable to set when it is called. Any other problem (an unknown name, bad options) is a
    configuration error now, like an unknown model provider.
    """
    from .credentials import MissingSecretError
    from .search.registry import create as create_search_provider
    from .tools.web import WebSearchTool

    name = config.search_provider
    if name is None:
        return WebSearchTool()
    try:
        provider = create_search_provider(name, config.search_providers.get(name, {}))
    except MissingSecretError as exc:
        return WebSearchTool(unavailable=f"search provider {name!r}: {exc}")
    return WebSearchTool(provider)


def _builtin_registry() -> ToolRegistry:
    from .tools.edit import EditTool
    from .tools.glob import GlobTool
    from .tools.grep import GrepTool
    from .tools.read import ReadTool
    from .tools.shell import ShellTool
    from .tools.write import WriteTool

    registry = ToolRegistry()
    registry.register("glob", GlobTool)
    registry.register("grep", GrepTool)
    registry.register("read", ReadTool)
    registry.register("write", WriteTool)
    registry.register("edit", EditTool)
    registry.register("shell", ShellTool)
    registry.register("web", _web_tool, configured=True)
    return registry


READ_ONLY_TOOLS: tuple[str, ...] = READ_ONLY_TOOL_NAMES
MUTATION_TOOLS: tuple[str, ...] = ("write", "edit", "shell")
DEFAULT_TOOLS: tuple[str, ...] = READ_ONLY_TOOLS + MUTATION_TOOLS

_registry: ToolRegistry | None = None


def builtin_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = _builtin_registry()
    return _registry
