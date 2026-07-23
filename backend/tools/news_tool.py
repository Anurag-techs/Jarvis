"""
JARVIS News Tool.

1. Why this module exists:
   Fulfills Version 1.0 requirement to fetch current headlines and news updates.

2. How it fits into the architecture:
   Implements `BaseTool`. Registered in `ToolRegistry`. Delegates to `backend.services.news_service`.

3. Which future modules will interact with it:
   - `backend.tools.registry.ToolRegistry`
   - `backend.services.news_service.NewsService`

4. Common mistakes to avoid:
   - Making non-cached, blocking external API calls during speech loops without timeouts.

5. Possible future improvements:
   - Category filtering (Tech, Finance, Politics) and RSS feed parsing.
"""

import logging
from typing import Any

from backend.services.news_service import NewsService
from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.news")


class NewsTool(BaseTool):
    """Tool for retrieving news headlines."""

    def __init__(self, news_service: NewsService | None = None) -> None:
        self._service = news_service or NewsService()

    @property
    def name(self) -> str:
        return "get_news"

    @property
    def description(self) -> str:
        return "Fetches recent news headlines and breaking news updates."

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        return "news" in lowered or "headlines" in lowered or "latest updates" in lowered

    def execute(self, **kwargs: Any) -> ToolResult:
        category = kwargs.get("category") or "general"
        logger.info("Handling news request for category: %s", category)

        news_report = self._service.get_top_headlines(category)
        return ToolResult(
            success=True,
            message=news_report,
            data={"category": category, "report": news_report},
        )
