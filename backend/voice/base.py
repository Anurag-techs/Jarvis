"""
JARVIS Voice Engine Abstract Interfaces.

1. Why this module exists:
   Defines clean contracts for Wake-Word Detection, Speech-to-Text (STT), and Text-to-Speech (TTS).
   Prevents hardcoding specific voice audio drivers or vendor libraries.

2. How it fits into the architecture:
   Part of the Voice abstraction layer. Concrete implementations (e.g. Whisper STT, Piper TTS, Porcupine WakeWord)
   implement these contracts.

3. Which future modules will interact with it:
   - `backend.voice.manager.VoiceManager`
   - Future concrete STT/TTS engine providers in Version 2.0.

4. Common mistakes to avoid:
   - Importing PyAudio, SpeechRecognition, or sounddevice directly into core modules.

5. Possible future improvements:
   - Asynchronous speech stream callbacks and voice activity detection (VAD) handles.
"""

from abc import ABC, abstractmethod


class BaseWakeWordDetector(ABC):
    """Abstract Base Class for wake-word detection engine."""

    @abstractmethod
    def listen_for_wake_word(self) -> bool:
        """Blocks until wake word is detected. Returns True when triggered."""


class BaseSpeechToText(ABC):
    """Abstract Base Class for Speech-To-Text (STT) transcription engine."""

    @abstractmethod
    def transcribe(self, audio_data: bytes | None = None) -> str:
        """Transcribes audio input into text string."""


class BaseTextToSpeech(ABC):
    """Abstract Base Class for Text-To-Speech (TTS) synthesis engine."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesizes text input into speech audio output."""
