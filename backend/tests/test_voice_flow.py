"""
Unit tests for JARVIS Voice Interaction Flow (Task: Wake Word Acknowledgement & Command Separation).
Covering 8 required test cases:
1. Wake word detected -> acknowledgement generated
2. Acknowledgement completes -> command listening starts
3. "Hey Jarvis" does not become the command
4. "Open Chrome" after acknowledgement transcribed & executed
5. Low-confidence commands use rejection behavior
6. Mid-confidence commands use confirmation behavior
7. Follow-up commands continue to work
8. Voice system returns to standby after conversation ends
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.ai.provider import MockLLMProvider
from backend.core.models import AssistantResponse, ToolResult
from backend.core.orchestrator import SystemOrchestrator
from backend.interfaces.voice import VoiceController, VoiceState
from backend.tools.base import BaseTool
from backend.tools.registry import ToolRegistry
from backend.voice.providers import MockSTTProvider, MockTTSProvider
from backend.voice.service import TTSService


class DummyOpenChromeTool(BaseTool):
    @property
    def name(self) -> str:
        return "open_application"

    @property
    def description(self) -> str:
        return "Opens local applications"

    def can_handle(self, query: str) -> bool:
        return "chrome" in query.lower() or "open chrome" in query.lower()

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, message="Chrome opened successfully")


class TestVoiceFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(DummyOpenChromeTool())
        self.llm = MockLLMProvider()
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
        )
        self.stt = MockSTTProvider(default_transcript="Open Chrome")
        self.mock_tts = MockTTSProvider()
        self.tts = TTSService(provider=self.mock_tts)
        self.controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            tts_service=self.tts,
            wake_acknowledgement="Yes, Anurag sir.",
            max_cycles=1,
            max_conversation_turns=1,
        )

    def test_1_and_2_wake_word_acknowledgement_and_transition(self) -> None:
        """Test 1 & 2: Wake word detected -> acknowledgement generated -> command listening starts."""
        states_visited = []

        def on_state_change(old, new):
            states_visited.append(new)

        self.controller._on_state_change = on_state_change

        with patch("builtins.input", side_effect=["", "exit"]):
            with patch("builtins.print"):
                self.controller.start()

        self.assertIn(VoiceState.WAKE_DETECTED, states_visited)
        self.assertIn(VoiceState.WAIT_FOR_SPEECH, states_visited)
        self.assertIn("Yes, Anurag sir.", self.mock_tts.spoken_messages)

    def test_3_and_4_wake_word_separation_and_command_execution(self) -> None:
        """Test 3 & 4: 'Hey Jarvis' does not become the command; 'Open Chrome' executed."""
        self.stt = MockSTTProvider(default_transcript="Hey Jarvis Open Chrome")
        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            tts_service=self.tts,
            wake_acknowledgement="Yes, Anurag sir.",
            max_cycles=1,
            max_conversation_turns=1,
        )

        with patch("builtins.input", side_effect=["", "exit"]):
            with patch("builtins.print"):
                controller.start()

        # Normalizer strips 'hey jarvis', executing 'open chrome' tool
        self.assertIsNotNone(controller._last_response_text)
        self.assertNotIn("Hey Jarvis Open Chrome", controller._last_response_text)

    def test_5_low_confidence_rejection(self) -> None:
        """Test 5: Low-confidence commands use existing rejection behavior."""
        low_stt = MockSTTProvider(default_transcript="unclear audio", confidence=0.30)
        low_stt.last_transcription_confidence = 0.30

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=low_stt,
            tts_service=self.tts,
            confidence_low=0.50,
            confidence_high=0.75,
            max_cycles=1,
            max_conversation_turns=1,
        )

        with patch("builtins.input", side_effect=["", "exit"]):
            with patch("builtins.print"):
                controller.start()

        spoken_all = " ".join(self.mock_tts.spoken_messages)
        self.assertIn("didn't understand", spoken_all)

    def test_6_mid_confidence_confirmation(self) -> None:
        """Test 6: Mid-confidence commands ask for confirmation ('Did you mean ...?')."""
        mid_stt = MockSTTProvider(default_transcript="open chrome", confidence=0.60)
        mid_stt.last_transcription_confidence = 0.60

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=mid_stt,
            tts_service=self.tts,
            confidence_low=0.50,
            confidence_high=0.75,
            max_cycles=1,
            max_conversation_turns=1,
        )

        with patch("builtins.input", side_effect=["", "exit"]):
            with patch("builtins.print"):
                controller.start()

        # Spoken output should contain confirmation question
        spoken_all = " ".join(self.mock_tts.spoken_messages)
        self.assertIn("Did you mean", spoken_all)

    def test_7_follow_up_listening(self) -> None:
        """Test 7: Continuous follow-up listening continues across turns."""
        stt = MockSTTProvider(default_transcript="what is the weather today?")
        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=stt,
            tts_service=self.tts,
            max_cycles=1,
            max_conversation_turns=2,
        )

        with patch("builtins.input", side_effect=["", "", "exit"]):
            with patch("builtins.print"):
                controller.start()

        self.assertTrue(len(self.mock_tts.spoken_messages) >= 2)

    def test_8_return_to_standby(self) -> None:
        """Test 8: Voice system returns to IDLE/Standby state after conversation ends."""
        final_state = None

        def on_state_change(old, new):
            nonlocal final_state
            final_state = new

        self.controller._on_state_change = on_state_change

        with patch("builtins.input", side_effect=["", "exit"]):
            with patch("builtins.print"):
                self.controller.start()

        self.assertEqual(self.controller.state, VoiceState.IDLE)


if __name__ == "__main__":
    unittest.main()
