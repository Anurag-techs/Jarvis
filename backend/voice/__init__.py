"""
JARVIS Voice Module Package Initialization.

1. Why this module exists:
   Exposes speech engine interfaces and VoiceManager for managing audio lifecycle.

2. How it fits into the architecture:
   Isolates voice hardware/software interfaces from core reasoning logic.

3. Which future modules will interact with it:
   - `backend.services.speech_service`
   - `backend.core.orchestrator`

4. Common mistakes to avoid:
   - Coupling speech recognition libraries directly into core orchestrator code.

5. Possible future improvements:
   - Streaming audio byte generators for real-time voice conversations.
"""

from backend.voice.base import BaseSpeechToText, BaseTextToSpeech, BaseWakeWordDetector
from backend.voice.manager import VoiceManager

__all__ = [
    "BaseSpeechToText",
    "BaseTextToSpeech",
    "BaseWakeWordDetector",
    "VoiceManager",
]
