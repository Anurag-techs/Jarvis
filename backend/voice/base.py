"""
JARVIS Voice Engine Abstract Interfaces.

1. Why this module exists:
   Defines clean contracts for Wake-Word Detection, Speech-to-Text (STT), and Text-to-Speech (TTS).
   Prevents hardcoding specific voice audio drivers or vendor libraries.

2. How it fits into the architecture:
   Part of the Voice abstraction layer. Concrete providers implement these contracts.

3. Which future modules will interact with it:
   - `backend.voice.service.TTSService`
   - STT Providers (`FasterWhisperProvider`, `MockSTTProvider`)
   - TTS Providers (`Pyttsx3Provider`, `MockTTSProvider`)

4. Common mistakes to avoid:
   - Importing engine SDKs directly into core modules.

5. Possible future improvements:
   - Asynchronous speech stream callbacks and voice activity detection (VAD) handles.
"""

from abc import ABC, abstractmethod


class SpeechToTextProvider(ABC):
    """Abstract Base Class for Speech-To-Text (STT) transcription engine providers."""

    @abstractmethod
    def listen_and_transcribe(self, duration: float = 5.0) -> str:
        """Captures audio from default microphone and returns transcribed text string.

        Args:
            duration: Listening duration in seconds.

        Returns:
            Transcribed text string.
        """

    @abstractmethod
    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        """Transcribes raw PCM / WAV audio byte stream into text string."""


class TextToSpeechProvider(ABC):
    """Abstract Base Class for Text-To-Speech (TTS) engine providers."""

    @abstractmethod
    def speak(self, text: str, interrupt: bool = False) -> None:
        """Synthesizes text into speech output.

        Args:
            text: Text string to speak.
            interrupt: If True, stop ongoing speech before speaking new text.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stops any ongoing speech synthesis output."""


class BaseWakeWordDetector(ABC):
    """Abstract Base Class for wake-word detection engine."""

    @abstractmethod
    def listen_for_wake_word(self) -> bool:
        """Blocks until wake word is detected. Returns True when triggered."""


class BaseSpeechToText(ABC):
    """Abstract Base Class for Speech-To-Text (STT) transcription engine (Legacy V1)."""

    @abstractmethod
    def transcribe(self, audio_data: bytes | None = None) -> str:
        """Transcribes audio input into text string."""


class BaseTextToSpeech(ABC):
    """Abstract Base Class for Text-To-Speech (TTS) synthesis engine (Legacy V1)."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesizes text input into speech audio output."""
