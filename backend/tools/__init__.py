"""
JARVIS Tools Package Initialization.

1. Why this module exists:
   Exposes tool contracts, ToolRegistry, and Version 1.0 executable system tools.

2. How it fits into the architecture:
   The tool layer encapsulates actionable capabilities invoked by the SystemOrchestrator.

3. Which future modules will interact with it:
   - `backend.core.orchestrator.SystemOrchestrator`
   - Future plugin loading infrastructure.

4. Common mistakes to avoid:
   - Importing concrete tools inside core orchestrator instead of registering them in ToolRegistry.

5. Possible future improvements:
   - Dynamic plugin directory auto-discovery and loading.
"""

from backend.tools.application_tool import ApplicationTool
from backend.tools.base import BaseTool, ToolResult
from backend.tools.browser_tool import BrowserTool
from backend.tools.news_tool import NewsTool
from backend.tools.registry import ToolRegistry
from backend.tools.screenshot_tool import ScreenshotTool
from backend.tools.search_tool import SearchTool
from backend.tools.system_tool import SystemTool
from backend.tools.weather_tool import WeatherTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "ApplicationTool",
    "BrowserTool",
    "NewsTool",
    "ScreenshotTool",
    "SearchTool",
    "SystemTool",
    "WeatherTool",
]
