"""
Unit tests for JARVIS Text-To-Speech (TTS) Foundation (Sprint 2.1).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from backend.core.exceptions import VoiceError
from backend.voice.base import TextToSpeechProvider
from backend.voice.providers import MockTTSProvider, Pyttsx3Provider
from backend.voice.service import TTSService


class TestTTSFoundation(unittest.TestCase):
    def test_mock_tts_provider_spoken_messages(self) -> None:
        """Verifies that MockTTSProvider records spoken text for assertions."""
        mock_provider = MockTTSProvider()
        mock_provider.speak("Hello world")
        mock_provider.speak("Second message", interrupt=True)

        self.assertEqual(len(mock_provider.spoken_messages), 2)
        self.assertEqual(mock_provider.spoken_messages[0], "Hello world")
        self.assertEqual(mock_provider.spoken_messages[1], "Second message")

    def test_mock_tts_provider_stop_and_clear(self) -> None:
        """Verifies stop and clear functionality on MockTTSProvider."""
        mock_provider = MockTTSProvider()
        mock_provider.speak("Test text")
        mock_provider.stop()
        self.assertTrue(mock_provider.is_stopped)

        mock_provider.clear()
        self.assertEqual(len(mock_provider.spoken_messages), 0)
        self.assertFalse(mock_provider.is_stopped)

    def test_tts_service_dependency_injection(self) -> None:
        """Verifies TTSService delegates speak and stop calls to injected provider."""
        mock_provider = MockTTSProvider()
        service = TTSService(provider=mock_provider)

        service.speak("Greetings JARVIS", interrupt=False)
        self.assertIn("Greetings JARVIS", mock_provider.spoken_messages)

        service.stop()
        self.assertTrue(mock_provider.is_stopped)

    def test_tts_service_extensible_speak_parameters(self) -> None:
        """Verifies TTSService supports optional parameters like interrupt and kwargs."""
        mock_provider = MockTTSProvider()
        service = TTSService(provider=mock_provider)

        service.speak("Message 1")
        service.speak("Message 2", interrupt=True, rate=200)

        self.assertEqual(len(mock_provider.spoken_messages), 2)

    def test_tts_service_skips_empty_strings(self) -> None:
        """Verifies TTSService skips empty or whitespace strings."""
        mock_provider = MockTTSProvider()
        service = TTSService(provider=mock_provider)

        service.speak("   ")
        service.speak("")
        self.assertEqual(len(mock_provider.spoken_messages), 0)

    def test_pyttsx3_provider_initialization(self) -> None:
        """Verifies Pyttsx3Provider initializes engine once via mocked module."""
        mock_pyttsx3 = MagicMock()
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch.dict(sys.modules, {"pyttsx3": mock_pyttsx3}):
            provider = Pyttsx3Provider()
            mock_pyttsx3.init.assert_called_once()

            provider.speak("Hello")
            mock_engine.say.assert_called_once_with("Hello")
            mock_engine.runAndWait.assert_called_once()

    def test_pyttsx3_provider_raises_voice_error(self) -> None:
        """Verifies Pyttsx3Provider raises VoiceError on init failure."""
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.side_effect = Exception("Audio driver unavailable")

        with patch.dict(sys.modules, {"pyttsx3": mock_pyttsx3}):
            with self.assertRaises(VoiceError) as ctx:
                Pyttsx3Provider()

            self.assertIn("Failed to initialize pyttsx3", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
