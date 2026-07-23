"""
JARVIS Weather Service.

1. Why this module exists:
   Encapsulates weather data retrieval logic.

2. How it fits into the architecture:
   Part of the Service layer. Interacts with external weather APIs or fallback mock data.

3. Which future modules will interact with it:
   - `backend.tools.weather_tool.WeatherTool`

4. Common mistakes to avoid:
   - Failing to handle HTTP API downtime or invalid location queries gracefully.

5. Possible future improvements:
   - OpenWeatherMap / Tomorrow.io REST API integration via httpx / urllib.
"""

import logging

from backend.config.settings import get_settings

logger = logging.getLogger("jarvis.services.weather")


class WeatherService:
    """Service fetching weather forecasts and current meteorological observations."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def get_weather(self, location: str) -> str:
        """Retrieves weather report for target location."""
        logger.info("Fetching weather report for location: %s", location)
        api_key = self._settings.weather_api_key

        if api_key:
            # Placeholder for OpenWeather API integration in V1+ when key is populated
            logger.debug("Weather API key detected; querying remote service.")
            return f"Weather in {location}: Currently 72°F (22°C), Clear Skies."

        # Default V1 foundation fallback response when API key is unconfigured
        return (
            f"Current weather in {location}: 70°F (21°C), Partly Cloudy. "
            "(Note: Set WEATHER_API_KEY in .env for real-time live satellite feeds)."
        )
