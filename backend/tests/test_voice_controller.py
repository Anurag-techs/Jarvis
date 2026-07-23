"""
Unit tests for JARVIS VoiceController (Task 2).
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.ai.provider import MockLLMProvider
from backend.core.exceptions import VoiceError
from backend.core.models import AssistantResponse
from backend.core.orchestrator import SystemOrchestrator
from backend.interfaces.voice import VoiceController
from backend.tools.registry import ToolRegistry
from backend.voice.providers import MockSTTProvider, MockTTSProvider
from backend.voice.service import TTSService


class TestVoiceController(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.llm = MockLLMProvider()
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
        )
        self.stt = MockSTTProvider(default_transcript="hello")
        self.mock_tts = MockTTSProvider()
        self.tts = TTSService(provider=self.mock_tts)
        self.controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            tts_service=self.tts,
            listening_duration=1.0,
        )

    def test_voice_controller_single_cycle_flow(self) -> None:
        """Verifies single voice interaction cycle: STT -> Orchestrator -> TTS."""
        with patch("builtins.input", side_effect=["", "exit"]):
            with patch("builtins.print"):
                self.controller.start()

                # Verify STT listening called
                self.assertEqual(len(self.stt.recorded_durations), 1)
                # Verify TTS spoken message stored
                self.assertIn("Hello! I am JARVIS.", self.mock_tts.spoken_messages)

    def test_voice_controller_empty_transcription_handling(self) -> None:
        """Verifies empty transcription triggers 'I didn't catch that.' fallback."""
        empty_stt = MockSTTProvider(default_transcript="")
        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=empty_stt,
            tts_service=self.tts,
        )

        with patch("builtins.input", side_effect=["", "exit"]):
            controller.start()
            self.assertIn("I didn't catch that.", self.mock_tts.spoken_messages)

    def test_voice_controller_microphone_exception_recovery(self) -> None:
        """Verifies VoiceController recovers gracefully when STT throws an error."""
        failing_stt = MagicMock()
        failing_stt.listen_and_transcribe.side_effect = VoiceError("Microphone device disconnected")

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=failing_stt,
            tts_service=self.tts,
        )

        with patch("builtins.input", side_effect=["", "exit"]):
            with patch("builtins.print") as mock_print:
                controller.start()  # Should not raise exception
                mock_print.assert_any_call("\n[Microphone Error]: Microphone device disconnected")

    def test_voice_controller_tts_exception_recovery(self) -> None:
        """Verifies VoiceController continues gracefully if TTS synthesis fails."""
        failing_tts = MagicMock()
        failing_tts.speak.side_effect = Exception("Audio driver lock")

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            tts_service=failing_tts,
        )

        with patch("builtins.input", side_effect=["", "exit"]):
            controller.start()  # Should complete cycle without raising exception


if __name__ == "__main__":
    unittest.main()
