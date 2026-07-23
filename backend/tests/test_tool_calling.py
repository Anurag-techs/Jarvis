"""
Unit tests for JARVIS Milestone 2 - AI Core + Tool Calling.
"""

import unittest
from unittest.mock import MagicMock

from backend.ai.provider import MockLLMProvider
from backend.core.models import AssistantResponse, ToolCall, ToolResult
from backend.core.orchestrator import SystemOrchestrator
from backend.tools.executor import ToolExecutor
from backend.tools.registry import ToolRegistry
from backend.tools.weather_tool import WeatherTool


class TestToolCallingPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(WeatherTool())
        self.executor = ToolExecutor(registry=self.registry)
        self.llm = MockLLMProvider()
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
            tool_executor=self.executor,
        )

    def test_tool_call_model_instantiation(self) -> None:
        """Verifies ToolCall model creation and serialization."""
        call = ToolCall(tool="open_application", arguments={"app_name": "chrome"})
        self.assertEqual(call.tool, "open_application")
        self.assertEqual(call.arguments.get("app_name"), "chrome")

    def test_tool_executor_valid_execution(self) -> None:
        """Verifies ToolExecutor validates tool presence and executes target tool cleanly."""
        call = ToolCall(tool="get_weather", arguments={"location": "Tokyo"})
        result = self.executor.execute_tool_call(call)
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.success)
        self.assertIn("Tokyo", result.message)

    def test_tool_executor_invalid_tool(self) -> None:
        """Verifies ToolExecutor captures invalid tool requests gracefully."""
        call = ToolCall(tool="non_existent_tool", arguments={})
        result = self.executor.execute_tool_call(call)
        self.assertFalse(result.success)
        self.assertIn("not available", result.message)

    def test_tool_registry_get_available_tools_metadata(self) -> None:
        """Verifies get_available_tools includes parameter schemas."""
        metadata = self.registry.get_available_tools()
        self.assertTrue(len(metadata) > 0)
        self.assertEqual(metadata[0]["name"], "get_weather")
        self.assertIn("parameters", metadata[0])

    def test_orchestrator_tool_calling_pipeline(self) -> None:
        """Verifies end-to-end tool calling flow from user input to tool execution response."""
        response = self.orchestrator.process("What is the weather in Paris?")
        self.assertIsInstance(response, AssistantResponse)
        self.assertTrue(response.success)
        self.assertTrue(len(response.tool_calls) > 0)
        self.assertEqual(response.tool_calls[0].tool, "get_weather")
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
