"""web: optional web search through a pluggable provider (disabled by default).

Kennel chooses no search provider by default. The ``search_provider`` config key selects
one from :mod:`kennel.search` (the registry builds this tool with it), or an application
passes ``WebSearchTool(my_provider)`` itself. Results keep source URL/title/snippet so the
agent can cite them. Enabling web access sends queries off the device.

The result text goes to the model only. ``metadata`` -- which reaches events, the session
log and ``--output-format`` -- carries the provider name and the number of results, never
their text (principle 3).
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolExecutionError
from ..permissions import PermissionKind
from ..search.base import SearchProvider, SearchProviderInfo, SearchResult
from .base import Tool, ToolContext, ToolParameter, ToolResult

__all__ = ["SearchProvider", "SearchResult", "WebSearchTool"]

NOT_CONFIGURED = "web search is not configured (no SearchProvider); answer from local files instead"


class WebSearchTool(Tool):
    name = "web"
    description = "Search the web and return titles, URLs and snippets. Only available when web access is enabled."
    permission = PermissionKind.WEB
    parameters = (
        ToolParameter("query", "string", "Search query"),
        ToolParameter("limit", "integer", "Maximum number of results (default: 5)", required=False),
    )

    def __init__(self, provider: SearchProvider | None = None, *, unavailable: str | None = None) -> None:
        """``unavailable`` explains why a configured provider could not be built (a missing key)."""
        self.provider = provider
        self.unavailable = unavailable

    @property
    def provider_info(self) -> SearchProviderInfo | None:
        info = getattr(self.provider, "info", None)
        return info if isinstance(info, SearchProviderInfo) else None

    def describe(self) -> str | None:
        """Where searches go, for the CLI header and ``Session.status()``; ``None`` when unset."""
        if self.unavailable is not None:
            return f"unavailable: {self.unavailable}"
        if self.provider is None:
            return None
        info = self.provider_info
        return info.describe() if info is not None else type(self.provider).__name__

    def summarize(self, arguments: dict[str, Any]) -> str:
        return f"WebSearch {arguments.get('query', '')!r}"

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if self.unavailable is not None:
            raise ToolExecutionError(f"web search is not available: {self.unavailable}; answer from local files instead")
        if self.provider is None:
            raise ToolExecutionError(NOT_CONFIGURED)
        limit = max(1, min(int(arguments.get("limit") or 5), 10))
        results = await self.provider.search(arguments["query"], limit=limit)
        info = self.provider_info
        metadata = {"provider": info.name if info is not None else type(self.provider).__name__, "returned": len(results)}
        if not results:
            return ToolResult(content="No results.", metadata=metadata)
        lines = [f"{i}. {r.title}\n   {r.url}\n   {r.snippet}" for i, r in enumerate(results, 1)]
        return ToolResult(content="\n".join(lines), metadata=metadata)
