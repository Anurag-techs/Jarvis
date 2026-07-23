"""
Unit tests for JARVIS Milestone 4.5 - Voice Stabilization.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from backend.ai.provider import MockLLMProvider
from backend.core.orchestrator import SystemOrchestrator
from backend.interfaces.voice import VoiceController
from backend.tools.registry import ToolRegistry
from backend.voice.providers import FasterWhisperProvider, MockSTTProvider, MockTTSProvider
from backend.voice.service import TTSService
from backend.voice.wake_word import MockWakeWordDetector, OpenWakeWordDetector


class TestVoiceStabilization(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.llm = MockLLMProvider()
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
        )
        self.mock_tts = MockTTSProvider()
        self.tts = TTSService(provider=self.mock_tts)

    def test_microphone_cooldown_execution(self) -> None:
        """Verifies cooldown delay is applied post-TTS speaking."""
        stt = MockSTTProvider(default_transcript="hello")
        wake_word = MockWakeWordDetector(trigger_sequence=[True])

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=stt,
            wake_word_detector=wake_word,
            tts_service=self.tts,
            listen_timeout=0.1,
            post_response_timeout=0.1,
            cooldown_seconds=0.01,
            max_cycles=1,
            max_conversation_turns=1,  # Bound inner loop: MockSTTProvider always returns "hello"
        )

        with patch("time.sleep") as mock_sleep:
            controller.start()
            # Verify cooldown sleep called with 0.01s
            mock_sleep.assert_any_call(0.01)


    def test_followup_silence_returns_immediately_to_standby(self) -> None:
        """Verifies follow-up mode exits immediately to Standby on silent transcript without repeating."""
        stt_mock = MagicMock()
        stt_mock.listen_and_transcribe.side_effect = ["hello", ""]  # 1st active speech, 2nd follow-up silence
        wake_word = MockWakeWordDetector(trigger_sequence=[True])

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=stt_mock,
            wake_word_detector=wake_word,
            tts_service=self.tts,
            listen_timeout=0.1,
            post_response_timeout=0.1,
            cooldown_seconds=0.0,
            max_cycles=1,
        )

        controller.start()
        # Ensure STT called exactly twice (active command + 1 silent follow-up before returning to Standby)
        self.assertEqual(stt_mock.listen_and_transcribe.call_count, 2)

    def test_stt_and_wake_word_overflow_recovery(self) -> None:
        """Verifies PyAudio stream reads wrap buffer overflow exceptions without crashing."""
        mock_pyaudio_module = MagicMock()
        mock_audio_instance = MagicMock()
        mock_stream_instance = MagicMock()

        mock_pyaudio_module.PyAudio.return_value = mock_audio_instance
        mock_pyaudio_module.paInt16 = 1
        mock_audio_instance.open.return_value = mock_stream_instance
        mock_audio_instance.get_sample_size.return_value = 2

        # Simulate buffer overflow IOError on stream.read
        mock_stream_instance.read.side_effect = [IOError("Input overflowed"), b"\x00" * 2560]

        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Hello"
        mock_whisper_module.WhisperModel.return_value = mock_model
        mock_model.transcribe.return_value = ([mock_segment], None)

        with patch.dict(sys.modules, {"pyaudio": mock_pyaudio_module, "faster_whisper": mock_whisper_module}):
            provider = FasterWhisperProvider(model_size="base", device="cpu")
            transcript = provider.listen_and_transcribe(duration=0.1)
            self.assertEqual(transcript, "Hello")


if __name__ == "__main__":
    unittest.main()
