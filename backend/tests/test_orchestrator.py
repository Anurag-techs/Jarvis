"""
Unit tests for JARVIS SystemOrchestrator (Sprint 2.2).
"""

import unittest
from backend.ai.provider import MockLLMProvider
from backend.core.models import AssistantResponse
from backend.core.orchestrator import SystemOrchestrator
from backend.tools.registry import ToolRegistry
from backend.tools.weather_tool import WeatherTool
from backend.voice.providers import MockTTSProvider
from backend.voice.service import TTSService


class TestOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(WeatherTool())
        self.mock_tts_provider = MockTTSProvider()
        self.tts_service = TTSService(provider=self.mock_tts_provider)
        self.llm = MockLLMProvider(model_name="test-v1")
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
            tts_service=self.tts_service,
        )

    def test_orchestrator_conversational_hello(self) -> None:
        """Verifies that 'Hello' maps to deterministic MockLLMProvider response."""
        response = self.orchestrator.process("Hello")
        self.assertIsInstance(response, AssistantResponse)
        self.assertTrue(response.success)
        self.assertEqual(response.text, "Hello! I am JARVIS.")
        self.assertIn("Hello! I am JARVIS.", self.mock_tts_provider.spoken_messages)

    def test_orchestrator_conversational_name(self) -> None:
        """Verifies that 'What is your name?' maps to name response."""
        response = self.orchestrator.process("What is your name?")
        self.assertTrue(response.success)
        self.assertEqual(response.text, "My name is JARVIS.")

    def test_orchestrator_conversational_fallback(self) -> None:
        """Verifies unknown prompt maps to default fallback response."""
        response = self.orchestrator.process("What is quantum entanglement?")
        self.assertTrue(response.success)
        self.assertEqual(response.text, "I'm still under development.")

    def test_orchestrator_tool_routing(self) -> None:
        """Verifies that tool keywords trigger tool execution and return AssistantResponse."""
        response = self.orchestrator.process("What is the weather in Tokyo?")
        self.assertTrue(response.success)
        self.assertIn("Tokyo", response.text)

    def test_orchestrator_empty_input(self) -> None:
        """Verifies handling of empty input string."""
        response = self.orchestrator.process("   ")
        self.assertFalse(response.success)
        self.assertIn("did not receive any input", response.text)


if __name__ == "__main__":
    unittest.main()
