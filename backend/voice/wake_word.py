"""
JARVIS Wake-Word Detection Engine.

1. Why this module exists:
   Provides hands-free background wake-word detection using `openWakeWord`.
   Monitors microphone audio stream with minimal CPU overhead until the wake phrase ("Hey Jarvis") is triggered.

2. How it fits into the architecture:
   Part of `backend.voice`. Injected into `VoiceController` interface.

3. Which future modules will interact with it:
   - `backend.interfaces.voice.VoiceController`

4. Common mistakes to avoid:
   - Running full Speech-to-Text models continuously in a tight loop (destroys CPU performance).
   - Blocking execution without catchable interrupt signals.

5. Possible future improvements:
   - Multi-phrase custom ONNX model loading.
"""

import logging
from typing import Any

from backend.core.exceptions import VoiceError
from backend.voice.base import BaseWakeWordDetector

logger = logging.getLogger("jarvis.voice.wake_word")


class OpenWakeWordDetector(BaseWakeWordDetector):
    """Hands-free background wake-word detector using openWakeWord."""

    def __init__(
        self,
        wake_word: str = "hey jarvis",
        sensitivity: float = 0.5,
    ) -> None:
        """Initializes openWakeWord model instance once.

        Args:
            wake_word: Wake phrase string.
            sensitivity: Detection threshold between 0.0 and 1.0.

        Raises:
            VoiceError: If openWakeWord fails to load.
        """
        self._wake_word = wake_word.lower()
        self._sensitivity = sensitivity

        try:
            import numpy as np
            import openwakeword
            from openwakeword.model import Model

            # Download or load openwakeword models
            openwakeword.utils.download_models()
            self._model = Model(inference_framework="onnx")
            logger.info("OpenWakeWordDetector initialized (Wake phrase: '%s', Sensitivity: %.2f)", wake_word, sensitivity)
        except Exception as exc:
            raise VoiceError(
                message=f"Failed to initialize openWakeWord detector: {exc}",
                details={"wake_word": wake_word},
            ) from exc

    def listen_for_wake_word(self) -> bool:
        """Streams small microphone audio chunks and monitors prediction score.

        Returns:
            True when wake word prediction score exceeds configured sensitivity threshold.

        Raises:
            VoiceError: If microphone stream fails.
        """
        logger.debug("Listening for wake phrase '%s'...", self._wake_word)
        try:
            import numpy as np
            import pyaudio

            chunk_size = 1280
            format_type = pyaudio.paInt16
            channels = 1
            rate = 16000

            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=format_type,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk_size,
            )

            try:
                while True:
                    try:
                        data = stream.read(chunk_size, exception_on_overflow=False)
                    except IOError as io_err:
                        logger.warning("PyAudio buffer overflow during wake word listening (recovered): %s", io_err)
                        continue

                    audio_chunk = np.frombuffer(data, dtype=np.int16)

                    # Predict score for current audio chunk
                    prediction = self._model.predict(audio_chunk)

                    for model_name, score in prediction.items():
                        if score >= self._sensitivity:
                            logger.info("Wake word detected! (Model: %s, Score: %.3f)", model_name, score)
                            self._model.reset()
                            return True

            finally:
                stream.stop_stream()
                stream.close()
                audio.terminate()

        except Exception as exc:
            logger.error("Error during wake word detection: %s", exc)
            raise VoiceError(
                message=f"Wake word microphone stream failed: {exc}",
                details={"wake_word": self._wake_word},
            ) from exc


class MockWakeWordDetector(BaseWakeWordDetector):
    """Mock Wake Word Detector returning pre-configured trigger sequences for testing."""

    def __init__(self, trigger_sequence: list[bool] | None = None) -> None:
        """Initialize mock detector.

        Args:
            trigger_sequence: List of boolean trigger return values per call.
        """
        self._sequence = trigger_sequence if trigger_sequence is not None else [True]
        self._call_count = 0

    def listen_for_wake_word(self) -> bool:
        """Simulates wake word listening and returns trigger boolean."""
        if not self._sequence:
            return False

        idx = min(self._call_count, len(self._sequence) - 1)
        triggered = self._sequence[idx]
        self._call_count += 1
        return triggered
