"""web: optional web search through a pluggable provider (disabled by default).

Kennel core ships no search provider. Applications that enable this tool must
supply a :class:`SearchProvider`; results keep source URL/title/snippet so the
agent can cite them. Enabling web access sends queries off the device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import ToolExecutionError
from ..permissions import PermissionKind
from .base import Tool, ToolContext, ToolParameter, ToolResult


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...


class WebSearchTool(Tool):
    name = "web"
    description = "Search the web and return titles, URLs and snippets. Only available when web access is enabled."
    permission = PermissionKind.WEB
    parameters = (
        ToolParameter("query", "string", "Search query"),
        ToolParameter("limit", "integer", "Maximum number of results (default: 5)", required=False),
    )

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self.provider = provider

    def summarize(self, arguments: dict[str, Any]) -> str:
        return f"WebSearch {arguments.get('query', '')!r}"

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if self.provider is None:
            raise ToolExecutionError("web search is not configured (no SearchProvider); answer from local files instead")
        limit = max(1, min(int(arguments.get("limit") or 5), 10))
        results = await self.provider.search(arguments["query"], limit=limit)
        if not results:
            return ToolResult(content="No results.", metadata={"results": []})
        lines = [f"{i}. {r.title}\n   {r.url}\n   {r.snippet}" for i, r in enumerate(results, 1)]
        return ToolResult(
            content="\n".join(lines),
            metadata={"results": [r.__dict__ for r in results]},
        )
