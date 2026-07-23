"""
JARVIS Services Package Initialization.

1. Why this module exists:
   Exposes high-level business capability services (LLM, Speech, System, Weather, News).

2. How it fits into the architecture:
   Services encapsulate domain business logic and bridge core orchestration with lower-level providers or external APIs.

3. Which future modules will interact with it:
   - `backend.core.orchestrator.SystemOrchestrator`
   - `backend.tools.*`

4. Common mistakes to avoid:
   - Mixing UI rendering logic or direct HTTP request handling inside service classes.

5. Possible future improvements:
   - Asynchronous batching and caching proxies for external services.
"""

from backend.services.llm_service import LLMService
from backend.services.news_service import NewsService
from backend.services.speech_service import SpeechService
from backend.services.system_service import SystemService
from backend.services.weather_service import WeatherService

__all__ = [
    "LLMService",
    "NewsService",
    "SpeechService",
    "SystemService",
    "WeatherService",
]
