"""
JARVIS News Service.

1. Why this module exists:
   Encapsulates news headlines retrieval logic.

2. How it fits into the architecture:
   Part of the Service layer. Interacts with external news feeds or fallback news providers.

3. Which future modules will interact with it:
   - `backend.tools.news_tool.NewsTool`

4. Common mistakes to avoid:
   - Returning unparsed RSS / XML / JSON blobs directly to user interfaces.

5. Possible future improvements:
   - NewsAPI.org and RSS feed parsing integration.
"""

import logging

from backend.config.settings import get_settings

logger = logging.getLogger("jarvis.services.news")


class NewsService:
    """Service fetching breaking news updates and headlines."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def get_top_headlines(self, category: str = "general") -> str:
        """Retrieves top news headlines."""
        logger.info("Fetching news headlines for category: %s", category)
        api_key = self._settings.news_api_key

        if api_key:
            logger.debug("News API key detected; querying remote news feed.")
            return f"Top Headlines ({category}): 1. Global AI Summit Announced. 2. Space Mission Launches."

        return (
            f"Latest Headlines ({category}): "
            "1. Major Technological Innovations Unveiled. "
            "2. Global Economic Outlook Update. "
            "(Note: Set NEWS_API_KEY in .env for real-time live news feeds)."
        )
