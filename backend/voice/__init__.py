"""
JARVIS Voice Package Initialization.

1. Why this module exists:
   Exposes speech interfaces, concrete providers, VoiceManager, TTSService, and WakeWord detectors.
"""

from backend.voice.base import (
    BaseSpeechToText,
    BaseTextToSpeech,
    BaseWakeWordDetector,
    SpeechToTextProvider,
    TextToSpeechProvider,
)
from backend.voice.manager import VoiceManager
from backend.voice.providers import (
    FasterWhisperProvider,
    MockSTTProvider,
    MockTTSProvider,
    Pyttsx3Provider,
)
from backend.voice.service import TTSService
from backend.voice.wake_word import MockWakeWordDetector, OpenWakeWordDetector

__all__ = [
    "SpeechToTextProvider",
    "TextToSpeechProvider",
    "BaseSpeechToText",
    "BaseTextToSpeech",
    "BaseWakeWordDetector",
    "VoiceManager",
    "Pyttsx3Provider",
    "MockTTSProvider",
    "FasterWhisperProvider",
    "MockSTTProvider",
    "OpenWakeWordDetector",
    "MockWakeWordDetector",
    "TTSService",
]
