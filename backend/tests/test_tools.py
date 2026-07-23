"""
Unit tests for JARVIS Tool Registry and V1.0 Foundation Tools.
"""

import unittest
from backend.tools.application_tool import ApplicationTool
from backend.tools.browser_tool import BrowserTool
from backend.tools.news_tool import NewsTool
from backend.tools.registry import ToolRegistry
from backend.tools.search_tool import SearchTool
from backend.tools.system_tool import SystemTool
from backend.tools.weather_tool import WeatherTool


class TestTools(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(ApplicationTool())
        self.registry.register(BrowserTool())
        self.registry.register(NewsTool())
        self.registry.register(SearchTool())
        self.registry.register(SystemTool())
        self.registry.register(WeatherTool())

    def test_tool_registry_registration(self) -> None:
        """Verifies tool registration and listing."""
        tools = self.registry.list_tools()
        self.assertIn("get_weather", tools)
        self.assertIn("open_application", tools)
        self.assertIn("open_website", tools)
        self.assertIn("system_control", tools)

    def test_weather_tool_execution(self) -> None:
        """Verifies execution of weather tool via registry."""
        result = self.registry.execute_tool("get_weather", {"location": "London"})
        self.assertTrue(result.success)
        self.assertIn("London", result.message)

    def test_unregistered_tool_execution(self) -> None:
        """Verifies error handling for unregistered tool execution."""
        result = self.registry.execute_tool("non_existent_tool", {})
        self.assertFalse(result.success)
        self.assertIn("not available", result.message)


if __name__ == "__main__":
    unittest.main()
