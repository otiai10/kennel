"""Registry of tool factories and resolution of tool specs into instances."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .errors import ConfigurationError
from .permissions import READ_ONLY_TOOL_NAMES
from .tools.base import Tool

ToolFactory = Callable[[], Tool]


class ToolRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ToolFactory] = {}

    def register(self, name: str, factory: ToolFactory) -> None:
        self._factories[name] = factory

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def create(self, name: str) -> Tool:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown tool {name!r}; available: {', '.join(self.names())}"
            ) from exc
        return factory()

    def resolve(self, specs: Iterable[str | Tool] | None, default: Iterable[str]) -> list[Tool]:
        """Turn names and/or instances into a list of unique tool instances."""
        items = list(default) if specs is None else list(specs)
        tools: list[Tool] = []
        seen: set[str] = set()
        for spec in items:
            tool = self.create(spec) if isinstance(spec, str) else spec
            if not isinstance(tool, Tool):
                raise ConfigurationError(f"Not a Tool: {spec!r}")
            if tool.name in seen:
                raise ConfigurationError(f"Duplicate tool name: {tool.name!r}")
            seen.add(tool.name)
            tools.append(tool)
        return tools


def _builtin_registry() -> ToolRegistry:
    from .tools.edit import EditTool
    from .tools.glob import GlobTool
    from .tools.grep import GrepTool
    from .tools.read import ReadTool
    from .tools.shell import ShellTool
    from .tools.web import WebSearchTool
    from .tools.write import WriteTool

    registry = ToolRegistry()
    registry.register("glob", GlobTool)
    registry.register("grep", GrepTool)
    registry.register("read", ReadTool)
    registry.register("write", WriteTool)
    registry.register("edit", EditTool)
    registry.register("shell", ShellTool)
    registry.register("web", WebSearchTool)
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
