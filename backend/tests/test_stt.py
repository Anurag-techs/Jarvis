"""
Unit tests for JARVIS Speech-To-Text (STT) Providers.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from backend.core.exceptions import VoiceError
from backend.voice.providers import FasterWhisperProvider, MockSTTProvider


class TestSTTProviders(unittest.TestCase):
    def test_mock_stt_provider_transcription(self) -> None:
        """Verifies MockSTTProvider returns pre-configured transcript string."""
        provider = MockSTTProvider(default_transcript="Open Chrome browser")
        result = provider.listen_and_transcribe(duration=3.0)

        self.assertEqual(result, "Open Chrome browser")
        self.assertEqual(len(provider.recorded_durations), 1)
        self.assertEqual(provider.recorded_durations[0], 3.0)

    def test_faster_whisper_provider_initialization_and_transcription(self) -> None:
        """Verifies FasterWhisperProvider initializes model once and transcribes mock audio."""
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = " Hello JARVIS "

        mock_whisper_module.WhisperModel.return_value = mock_model
        mock_model.transcribe.return_value = ([mock_segment], None)

        with patch.dict(sys.modules, {"faster_whisper": mock_whisper_module}):
            provider = FasterWhisperProvider(model_size="base", device="cpu")
            mock_whisper_module.WhisperModel.assert_called_once_with(
                model_size_or_path="base", device="cpu", compute_type="int8"
            )

            transcript = provider.transcribe_audio_bytes(b"dummy_wav_pcm_header_and_payload")
            self.assertEqual(transcript, "Hello JARVIS")

    def test_faster_whisper_provider_raises_voice_error_on_init_failure(self) -> None:
        """Verifies FasterWhisperProvider wraps SDK initialization errors in VoiceError."""
        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.side_effect = Exception("CUDA error or missing model file")

        with patch.dict(sys.modules, {"faster_whisper": mock_whisper_module}):
            with self.assertRaises(VoiceError) as ctx:
                FasterWhisperProvider(model_size="invalid-model")

            self.assertIn("Failed to initialize faster-whisper", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
