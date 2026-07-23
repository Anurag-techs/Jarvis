"""
JARVIS Voice Providers Package Initialization.

1. Why this module exists:
   Exposes concrete TTS and STT voice providers.
"""

from backend.voice.providers.faster_whisper_provider import FasterWhisperProvider
from backend.voice.providers.mock_provider import MockTTSProvider
from backend.voice.providers.mock_stt_provider import MockSTTProvider
from backend.voice.providers.pyttsx3_provider import Pyttsx3Provider

__all__ = [
    "Pyttsx3Provider",
    "MockTTSProvider",
    "FasterWhisperProvider",
    "MockSTTProvider",
]
