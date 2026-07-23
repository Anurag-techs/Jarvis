"""
Unit tests for JARVIS Milestone 4 - Wake Word & Continuous Listening.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from backend.ai.provider import MockLLMProvider
from backend.core.exceptions import VoiceError
from backend.core.orchestrator import SystemOrchestrator
from backend.interfaces.voice import VoiceController
from backend.tools.registry import ToolRegistry
from backend.voice.providers import MockSTTProvider, MockTTSProvider
from backend.voice.service import TTSService
from backend.voice.wake_word import MockWakeWordDetector, OpenWakeWordDetector


class TestWakeWordAndContinuousListening(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.llm = MockLLMProvider()
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
        )
        self.mock_tts = MockTTSProvider()
        self.tts = TTSService(provider=self.mock_tts)

    def test_mock_wake_word_detector_trigger_sequence(self) -> None:
        """Verifies MockWakeWordDetector returns configured trigger sequence."""
        detector = MockWakeWordDetector(trigger_sequence=[True, False, True])
        self.assertTrue(detector.listen_for_wake_word())
        self.assertFalse(detector.listen_for_wake_word())
        self.assertTrue(detector.listen_for_wake_word())

    def test_open_wakeword_detector_initialization_failure(self) -> None:
        """Verifies OpenWakeWordDetector wraps SDK initialization errors in VoiceError."""
        mock_openwakeword = MagicMock()
        mock_openwakeword.model.Model.side_effect = Exception("Model ONNX file missing")

        with patch.dict(sys.modules, {"openwakeword": mock_openwakeword, "openwakeword.model": mock_openwakeword.model}):
            with self.assertRaises(VoiceError) as ctx:
                OpenWakeWordDetector(wake_word="hey jarvis")

            self.assertIn("Failed to initialize openWakeWord detector", ctx.exception.message)

    def test_voice_controller_handsfree_wake_word_pipeline(self) -> None:
        """Verifies hands-free wake word -> active speech -> follow-up silence -> standby state transitions."""
        stt = MockSTTProvider(default_transcript="hello")
        wake_word = MockWakeWordDetector(trigger_sequence=[True])

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=stt,
            wake_word_detector=wake_word,
            tts_service=self.tts,
            listen_timeout=1.0,
            post_response_timeout=1.0,
            max_cycles=1,
        )

        controller.start()
        # Verify TTS spoken completion response
        self.assertIn("Hello! I am JARVIS.", self.mock_tts.spoken_messages)

    def test_voice_controller_followup_continuous_conversation(self) -> None:
        """Verifies continuous follow-up window processes consecutive speech commands without re-triggering wake word."""
        # STT returns speech on 1st call, follow-up speech on 2nd call, empty on 3rd call
        stt_mock = MagicMock()
        stt_mock.listen_and_transcribe.side_effect = ["hello", "what is your name", ""]
        wake_word = MockWakeWordDetector(trigger_sequence=[True])

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=stt_mock,
            wake_word_detector=wake_word,
            tts_service=self.tts,
            listen_timeout=1.0,
            post_response_timeout=1.0,
            max_cycles=1,
        )

        controller.start()
        self.assertIn("Hello! I am JARVIS.", self.mock_tts.spoken_messages)
        self.assertIn("My name is JARVIS.", self.mock_tts.spoken_messages)

    def test_voice_controller_recovers_from_wake_word_error(self) -> None:
        """Verifies state machine recovers gracefully when wake word detector encounters a driver error."""
        failing_detector = MagicMock()
        failing_detector.listen_for_wake_word.side_effect = VoiceError("Stream read overflow")

        stt = MockSTTProvider()
        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=stt,
            wake_word_detector=failing_detector,
            tts_service=self.tts,
            max_cycles=2,
        )

        with patch("builtins.print") as mock_print:
            controller.start()  # Should not raise exception
            mock_print.assert_any_call("\n[Wake Word Error]: Stream read overflow")


if __name__ == "__main__":
    unittest.main()
