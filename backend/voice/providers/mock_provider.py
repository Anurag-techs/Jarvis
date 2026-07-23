"""
JARVIS Mock TTS Provider.

1. Why this module exists:
   Provides a lightweight mock TTS provider for testing, headless environments, and fallback handling.

2. How it fits into the architecture:
   Concrete provider subclassing `TextToSpeechProvider`. Allows inspection of `spoken_messages` in tests.

3. Which future modules will interact with it:
   - `backend.tests.test_tts`
   - `backend.voice.service.TTSService`

4. Common mistakes to avoid:
   - Performing actual audio hardware output inside mock testing implementations.

5. Possible future improvements:
   - Simulated speech latency delays for async testing.
"""

from backend.voice.base import TextToSpeechProvider


class MockTTSProvider(TextToSpeechProvider):
    """Mock TTS Provider recording spoken text for test assertions."""

    def __init__(self) -> None:
        self.spoken_messages: list[str] = []
        self.is_stopped: bool = False

    def speak(self, text: str, interrupt: bool = False) -> None:
        """Records spoken text in `spoken_messages` list."""
        if interrupt:
            self.stop()
        self.spoken_messages.append(text)
        self.is_stopped = False

    def stop(self) -> None:
        """Sets `is_stopped` flag to True."""
        self.is_stopped = True

    def clear(self) -> None:
        """Clears test call history."""
        self.spoken_messages.clear()
        self.is_stopped = False
