"""
JARVIS Pyttsx3 TTS Provider.

1. Why this module exists:
   Provides offline cross-platform Text-To-Speech synthesis using the `pyttsx3` engine.

2. How it fits into the architecture:
   Concrete provider subclassing `TextToSpeechProvider`. Instantiated and injected into `TTSService`.

3. Which future modules will interact with it:
   - `backend.voice.service.TTSService`

4. Common mistakes to avoid:
   - Re-instantiating `pyttsx3.init()` on every `speak()` call (causes memory leaks & audio driver crashes).
   - Logging inside provider instead of raising `VoiceError` for higher layer recovery.

5. Possible future improvements:
   - Asynchronous thread runner for non-blocking `say()` and `runAndWait()` calls.
"""

from backend.core.exceptions import VoiceError
from backend.voice.base import TextToSpeechProvider


class Pyttsx3Provider(TextToSpeechProvider):
    """Offline TTS Provider leveraging pyttsx3."""

    def __init__(self) -> None:
        """Initializes the pyttsx3 engine instance once."""
        try:
            import pyttsx3  # Deferred import to handle missing dependency cleanly

            self._engine = pyttsx3.init()
        except Exception as exc:
            raise VoiceError(
                message=f"Failed to initialize pyttsx3 engine: {exc}",
                details={"provider": "pyttsx3"},
            ) from exc

    def speak(self, text: str, interrupt: bool = False) -> None:
        """Synthesizes text input to speech audio output.

        Args:
            text: Text string to speak.
            interrupt: If True, stop ongoing speech before speaking.

        Raises:
            VoiceError: If speech synthesis fails.
        """
        if not text or not text.strip():
            return

        try:
            if interrupt:
                self.stop()

            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            raise VoiceError(
                message=f"Pyttsx3 speech synthesis failed: {exc}",
                details={"text": text, "interrupt": interrupt},
            ) from exc

    def stop(self) -> None:
        """Stops any active speech output."""
        try:
            self._engine.stop()
        except Exception as exc:
            raise VoiceError(
                message=f"Failed to stop pyttsx3 engine: {exc}"
            ) from exc
