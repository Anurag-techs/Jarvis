"""
JARVIS Web Search Tool.

1. Why this module exists:
   Fulfills Version 1.0 requirement for general internet search queries.

2. How it fits into the architecture:
   Implements `BaseTool`. Registered in `ToolRegistry`. Opens web browser search query.

3. Which future modules will interact with it:
   - `backend.tools.registry.ToolRegistry`
   - Future Deep Research Agent (Version 4.0).

4. Common mistakes to avoid:
   - Scraping search result HTML directly using fragile DOM selectors in V1 foundation.

5. Possible future improvements:
   - Serper / DuckDuckGo API search integration for structured JSON search result retrieval.
"""

import logging
import urllib.parse
import webbrowser
from typing import Any

from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.search")


class SearchTool(BaseTool):
    """Tool for performing general web searches."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Performs an internet search query in the default browser."

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        return "search for" in lowered or "google " in lowered or "search " in lowered

    def execute(self, **kwargs: Any) -> ToolResult:
        search_query = kwargs.get("search_query") or kwargs.get("query", "").replace("search for", "").replace("search", "").strip()

        if not search_query:
            return ToolResult(
                success=False,
                message="Please provide a search term.",
                error="Missing search_query parameter",
            )

        encoded_query = urllib.parse.quote_plus(search_query)
        search_url = f"https://www.google.com/search?q={encoded_query}"
        logger.info("Executing web search for query: '%s' -> %s", search_query, search_url)

        try:
            webbrowser.open(search_url)
            return ToolResult(
                success=True,
                message=f"Searching the web for '{search_query}'.",
                data={"query": search_query, "url": search_url},
            )
        except Exception as exc:
            logger.error("Failed to execute search: %s", exc)
            return ToolResult(
                success=False,
                message=f"Could not execute web search for '{search_query}'.",
                error=str(exc),
            )
