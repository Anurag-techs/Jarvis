"""
JARVIS Voice Manager.

1. Why this module exists:
   Coordinates the voice lifecycle (Wake word -> Listen/STT -> Core processing -> TTS output).

2. How it fits into the architecture:
   Part of the Voice layer. Injected into `SystemOrchestrator` to enable voice input/output channels.

3. Which future modules will interact with it:
   - `backend.core.orchestrator.SystemOrchestrator`
   - `backend.services.speech_service`

4. Common mistakes to avoid:
   - Performing speech synthesis or recognition on the main thread in a blocking manner during async loops.

5. Possible future improvements:
   - Non-blocking async event loop integration with continuous wake-word listening.
"""

import logging

from backend.voice.base import BaseSpeechToText, BaseTextToSpeech, BaseWakeWordDetector

logger = logging.getLogger("jarvis.voice.manager")


class MockSpeechToText(BaseSpeechToText):
    """V1.0 Mock STT implementation."""

    def transcribe(self, audio_data: bytes | None = None) -> str:
        logger.debug("MockSTT: Transcribing default input")
        return "Hello JARVIS"


class MockTextToSpeech(BaseTextToSpeech):
    """V1.0 Mock TTS implementation using log output."""

    def speak(self, text: str) -> None:
        logger.info("[JARVIS VOICE OUTPUT]: %s", text)


class MockWakeWordDetector(BaseWakeWordDetector):
    """V1.0 Mock WakeWord implementation."""

    def listen_for_wake_word(self) -> bool:
        logger.debug("MockWakeWord: Triggered")
        return True


class VoiceManager:
    """Lifecycle manager coordinating STT, TTS, and Wake-Word engines."""

    def __init__(
        self,
        stt_engine: BaseSpeechToText | None = None,
        tts_engine: BaseTextToSpeech | None = None,
        wake_word_engine: BaseWakeWordDetector | None = None,
    ) -> None:
        self.stt = stt_engine or MockSpeechToText()
        self.tts = tts_engine or MockTextToSpeech()
        self.wake_word = wake_word_engine or MockWakeWordDetector()
        logger.info("VoiceManager initialized with STT/TTS components.")

    def listen(self) -> str:
        """Captures user audio input and returns transcribed text."""
        logger.info("Listening for speech input...")
        return self.stt.transcribe()

    def speak(self, text: str) -> None:
        """Outputs text as spoken audio output."""
        self.tts.speak(text)
