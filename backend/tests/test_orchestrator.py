"""
Unit tests for JARVIS SystemOrchestrator.
"""

import unittest
from backend.ai.provider import MockLLMProvider
from backend.core.orchestrator import SystemOrchestrator
from backend.tools.registry import ToolRegistry
from backend.tools.weather_tool import WeatherTool


class TestOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(WeatherTool())
        self.llm = MockLLMProvider(model_name="test-v1")
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
        )

    def test_orchestrator_conversational_input(self) -> None:
        """Verifies that generic conversational input routes to LLM provider."""
        result = self.orchestrator.process_command("Hello, how are you?")
        self.assertTrue(result.success)
        self.assertIn("JARVIS V1.0 Foundation", result.response_text)

    def test_orchestrator_tool_routing(self) -> None:
        """Verifies that tool keywords trigger tool execution."""
        result = self.orchestrator.process_command("What is the weather in Tokyo?")
        self.assertTrue(result.success)
        self.assertIn("Tokyo", result.response_text)

    def test_orchestrator_empty_input(self) -> None:
        """Verifies handling of empty input string."""
        result = self.orchestrator.process_command("   ")
        self.assertFalse(result.success)
        self.assertIn("did not receive any input", result.response_text)


if __name__ == "__main__":
    unittest.main()
