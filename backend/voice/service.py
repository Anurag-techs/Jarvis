"""
JARVIS Text-To-Speech (TTS) Service.

1. Why this module exists:
   Encapsulates text-to-speech service capability. Mediates between the core orchestrator/application
   and underlying `TextToSpeechProvider` instances via Dependency Injection.

2. How it fits into the architecture:
   Service layer in `backend.voice`. High-level core components call `TTSService` without knowing
   whether pyttsx3, ElevenLabs, or a mock engine is active.

3. Which future modules will interact with it:
   - `backend.core.startup.StartupManager`
   - `backend.core.orchestrator.SystemOrchestrator`
   - `backend.api.router` (WebSocket audio stream handlers in V2.0).

4. Common mistakes to avoid:
   - Coupling this service directly to `pyttsx3` or any specific vendor library.

5. Possible future improvements:
   - Voice queue management and asynchronous audio buffering.
"""

import logging
from typing import Any

from backend.voice.base import TextToSpeechProvider

logger = logging.getLogger("jarvis.voice.service")


class TTSService:
    """Service providing speech output capabilities using an injected TextToSpeechProvider."""

    def __init__(self, provider: TextToSpeechProvider) -> None:
        """Initialize service via constructor dependency injection.

        Args:
            provider: Concrete implementation subclassing TextToSpeechProvider.
        """
        self._provider = provider
        logger.info("TTSService initialized with provider: %s", type(provider).__name__)

    def speak(self, text: str, interrupt: bool = False, **kwargs: Any) -> None:
        """Synthesizes text into spoken output.

        Args:
            text: Text string to speak.
            interrupt: If True, stop ongoing speech output before speaking.
            **kwargs: Extensible options reserved for future provider parameters (e.g. rate, pitch).
        """
        if not text or not text.strip():
            logger.debug("TTSService: Skipping empty text input")
            return

        logger.debug("TTSService delegating speak (interrupt=%s, text='%s')", interrupt, text)
        self._provider.speak(text, interrupt=interrupt)

    def stop(self) -> None:
        """Stops any active speech output on the underlying provider."""
        logger.debug("TTSService delegating stop call to provider")
        self._provider.stop()
