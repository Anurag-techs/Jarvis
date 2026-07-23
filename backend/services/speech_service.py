"""
JARVIS Speech Service.

1. Why this module exists:
   Encapsulates high-level speech service capabilities (STT listening and TTS speaking).

2. How it fits into the architecture:
   Part of the Service layer. Bridges audio hardware managers (`VoiceManager`) with higher-level application logic.

3. Which future modules will interact with it:
   - `backend.core.orchestrator.SystemOrchestrator`
   - `backend.api.router` (Voice WebSocket stream endpoints in V2.0).

4. Common mistakes to avoid:
   - Mixing speech audio processing with text parsing logic.

5. Possible future improvements:
   - Voice cloning and audio emotion synthesis controls.
"""

import logging

from backend.voice.manager import VoiceManager

logger = logging.getLogger("jarvis.services.speech")


class SpeechService:
    """Service providing speech input/output capabilities."""

    def __init__(self, voice_manager: VoiceManager | None = None) -> None:
        self._voice_manager = voice_manager or VoiceManager()

    def listen_and_transcribe(self) -> str:
        """Triggers STT listening and returns text transcript."""
        logger.debug("SpeechService: initiating audio capture")
        return self._voice_manager.listen()

    def speak_text(self, text: str) -> None:
        """Triggers TTS synthesis for given response text."""
        logger.debug("SpeechService: synthesizing text response")
        self._voice_manager.speak(text)
