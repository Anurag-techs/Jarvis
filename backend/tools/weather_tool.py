"""
JARVIS Weather Tool.

1. Why this module exists:
   Fulfills Version 1.0 requirement to handle weather queries.

2. How it fits into the architecture:
   Implements `BaseTool`. Registered in `ToolRegistry`. Delegates to `backend.services.weather_service`.

3. Which future modules will interact with it:
   - `backend.tools.registry.ToolRegistry`
   - `backend.services.weather_service.WeatherService`

4. Common mistakes to avoid:
   - Coupling HTTP API request logic directly into the tool class instead of using a service layer.

5. Possible future improvements:
   - Location auto-detection based on IP / GPS metadata.
"""

import logging
from typing import Any

from backend.services.weather_service import WeatherService
from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.weather")


class WeatherTool(BaseTool):
    """Tool for retrieving current weather information."""

    def __init__(self, weather_service: WeatherService | None = None) -> None:
        self._service = weather_service or WeatherService()

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Retrieves weather conditions for a specified location."

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        return "weather" in lowered or "temperature" in lowered or "forecast" in lowered

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        location = kwargs.get("location")
        if not location and "in " in query.lower():
            idx = query.lower().find("in ")
            location = query[idx + 3:].strip("? .!")
        if not location:
            location = "New York"

        logger.info("Handling weather request for location: %s", location)

        weather_info = self._service.get_weather(location)
        return ToolResult(
            success=True,
            message=weather_info,
            data={"location": location, "report": weather_info},
        )
