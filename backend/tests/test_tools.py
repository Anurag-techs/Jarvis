"""
Unit tests for JARVIS Tool Registry and V1.0 Foundation Tools.
"""

import unittest
from unittest.mock import patch
from backend.tools.application_tool import ApplicationTool, _normalize_app_name
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


class TestAppNameNormalization(unittest.TestCase):
    """Regression tests for application name normalization layer."""

    def test_open_calculator_normalizes_to_calc(self) -> None:
        """'Open Calculator' must normalize to the 'calc' Windows executable."""
        self.assertEqual(_normalize_app_name("Open Calculator"), "calc")

    def test_calculator_normalizes_to_calc(self) -> None:
        """'Calculator' alone must normalize to 'calc'."""
        self.assertEqual(_normalize_app_name("Calculator"), "calc")

    def test_open_notepad_normalizes_to_notepad(self) -> None:
        """'Open Notepad' must normalize to 'notepad'."""
        self.assertEqual(_normalize_app_name("Open Notepad"), "notepad")

    def test_notepad_normalizes_to_notepad(self) -> None:
        """'Notepad' alone must normalize to 'notepad'."""
        self.assertEqual(_normalize_app_name("Notepad"), "notepad")

    def test_open_paint_normalizes_to_mspaint(self) -> None:
        """'Open Paint' must normalize to 'mspaint'."""
        self.assertEqual(_normalize_app_name("Open Paint"), "mspaint")

    def test_chrome_normalizes_to_chrome(self) -> None:
        """'Chrome' must normalize to 'chrome'."""
        self.assertEqual(_normalize_app_name("Chrome"), "chrome")

    def test_launch_prefix_stripped(self) -> None:
        """'Launch Calculator' must also normalize correctly."""
        self.assertEqual(_normalize_app_name("Launch Calculator"), "calc")

    def test_unknown_app_returns_stripped_name(self) -> None:
        """An unknown app should return the name stripped of the action verb."""
        self.assertEqual(_normalize_app_name("Open SomeUnknownApp"), "SomeUnknownApp")


class TestApplicationToolExecution(unittest.TestCase):
    """Regression tests for ApplicationTool execution path."""

    def setUp(self) -> None:
        self.tool = ApplicationTool()

    def test_open_calculator_resolves_correct_executable(self) -> None:
        """Verifies 'Open Calculator' is normalized to 'calc' before OS launch."""
        with patch("backend.tools.application_tool.run_system_command", return_value=(True, "")) as mock_cmd:
            result = self.tool.execute(app_name="Open Calculator")
            self.assertTrue(result.success)
            # The normalized executable 'calc' must appear in the OS command, not 'Open Calculator'
            called_cmd = mock_cmd.call_args[0][0]
            self.assertIn("calc", called_cmd)
            self.assertNotIn("Open Calculator", called_cmd)

    def test_open_notepad_resolves_correct_executable(self) -> None:
        """Verifies 'Open Notepad' is normalized to 'notepad' before OS launch."""
        with patch("backend.tools.application_tool.run_system_command", return_value=(True, "")) as mock_cmd:
            result = self.tool.execute(app_name="Open Notepad")
            self.assertTrue(result.success)
            called_cmd = mock_cmd.call_args[0][0]
            self.assertIn("notepad", called_cmd)

    def test_failed_application_launch_returns_failure_result(self) -> None:
        """Verifies ApplicationTool returns success=False on OS launch failure."""
        with patch("backend.tools.application_tool.run_system_command", return_value=(False, "File not found")):
            result = self.tool.execute(app_name="Open Calculator")
            self.assertFalse(result.success)
            # The error message must mention the original name, not raw internal details
            self.assertIn("Calculator", result.message)
            # Internal OS error must not be surfaced directly in the user-facing message
            self.assertNotIn("File not found", result.message)


if __name__ == "__main__":
    unittest.main()
