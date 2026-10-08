"""
JARVIS Mock STT Provider.

1. Why this module exists:
   Provides a lightweight mock Speech-To-Text provider for testing, headless environments, and fallback handling.

2. How it fits into the architecture:
   Concrete provider subclassing `SpeechToTextProvider`.

3. Which future modules will interact with it:
   - `backend.tests.test_stt`
   - `backend.core.startup.StartupManager`

4. Common mistakes to avoid:
   - Requiring physical microphone hardware during unit test execution.
"""

from backend.voice.base import SpeechToTextProvider


class MockSTTProvider(SpeechToTextProvider):
    """Mock STT Provider returning pre-configured transcription text."""

    def __init__(self, default_transcript: str = "What is the weather in London?", confidence: float = 1.0) -> None:
        self.last_transcription_confidence = confidence
        self.default_transcript = default_transcript
        self.recorded_durations: list[float] = []

    def listen_and_transcribe(self, duration: float = 5.0) -> str:
        """Simulates microphone capture and returns preset transcript string."""
        self.recorded_durations.append(duration)
        return self.default_transcript

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        """Simulates audio byte stream transcription."""
        self.recorded_durations.append(5.0)
        return self.default_transcript

