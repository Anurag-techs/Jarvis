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
    """Hands-free background wake-word detector using openWakeWord with advanced trigger logic."""

    def __init__(
        self,
        wake_word: str = "hey jarvis",
        sensitivity: float = 0.35,
        rolling_window_size: int = 5,
        min_consecutive_detections: int = 3,
        cooldown_duration: float = 1.0,
        lower_threshold: float = 0.2,
    ) -> None:
        """Initializes openWakeWord model instance once.

        Args:
            wake_word: Wake phrase string.
            sensitivity: Detection threshold between 0.0 and 1.0.
            rolling_window_size: Size of the rolling score history window.
            min_consecutive_detections: Number of consecutive scores above lower_threshold required.
            cooldown_duration: Duration in seconds to ignore triggers post-activation.
            lower_threshold: Lower score threshold for consecutive checks.

        Raises:
            VoiceError: If openWakeWord fails to load.
        """
        self._wake_word = wake_word.lower()
        self._sensitivity = sensitivity
        self._rolling_window_size = rolling_window_size
        self._min_consecutive_detections = min_consecutive_detections
        self._cooldown_duration = cooldown_duration
        self._lower_threshold = lower_threshold

        from collections import deque
        self._score_history = deque(maxlen=self._rolling_window_size)
        self._background_scores = []
        self._last_trigger_time = 0.0

        try:
            import numpy as np
            import openwakeword
            from openwakeword.model import Model

            # Download or load openwakeword models
            openwakeword.utils.download_models()

            # Map the configured wake word to the expected pre-trained model name
            model_name = self._wake_word.replace(" ", "_")
            self._model = Model(wakeword_models=[model_name], inference_framework="onnx")

            logger.info(
                "OpenWakeWordDetector initialized (Wake phrase: '%s', Sensitivity: %.2f, Window: %d, Min Consecutive: %d, Cooldown: %.1f s, Lower Threshold: %.2f)",
                wake_word,
                sensitivity,
                rolling_window_size,
                min_consecutive_detections,
                cooldown_duration,
                lower_threshold,
            )
            logger.info("Loaded models: %s", list(self._model.models.keys()))
        except Exception as exc:
            raise VoiceError(
                message=f"Failed to initialize openWakeWord detector: {exc}",
                details={"wake_word": wake_word},
            ) from exc

    def listen_for_wake_word(self, check_running: Any = None, stream: Any = None) -> bool:
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
            import time

            chunk_size = 1280
            format_type = pyaudio.paInt16
            channels = 1
            rate = 16000

            if stream is None:
                audio = pyaudio.PyAudio()

                # Retrieve default input device info for logging
                try:
                    device_info = audio.get_default_input_device_info()
                    device_index = device_info.get("index")
                    device_name = device_info.get("name")
                except Exception as e:
                    device_index = "unknown"
                    device_name = f"unknown ({e})"

                logger.info(
                    "Microphone settings: device_index=%s, device_name=%r, sample_rate=%d, chunk_size=%d",
                    device_index,
                    device_name,
                    rate,
                    chunk_size,
                )

                stream = audio.open(
                    format=format_type,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk_size,
                )
                own_stream = True
            else:
                audio = None
                own_stream = False

            try:
                last_log_time = 0.0
                while True:
                    if check_running and not check_running():
                        logger.info("Shutdown requested. Exiting listen_for_wake_word.")
                        return False
                    try:
                        data = stream.read(chunk_size, exception_on_overflow=False)
                    except IOError as io_err:
                        logger.warning("PyAudio buffer overflow during wake word listening (recovered): %s", io_err)
                        continue

                    audio_chunk = np.frombuffer(data, dtype=np.int16)

                    # Predict score for current audio chunk
                    prediction = self._model.predict(audio_chunk)

                    model_name = self._wake_word.replace(" ", "_")
                    score = prediction.get(model_name, 0.0)

                    # 1. Update background scores if not triggered
                    if score < self._lower_threshold:
                        self._background_scores.append(score)
                        if len(self._background_scores) > 100:
                            self._background_scores.pop(0)

                    # 2. Compute tuned sensitivity
                    if self._background_scores:
                        bg_mean = float(np.mean(self._background_scores))
                        bg_std = float(np.std(self._background_scores))
                        tuned_sensitivity = max(self._sensitivity, bg_mean + 3 * bg_std + 0.1)
                        tuned_sensitivity = min(tuned_sensitivity, 0.8)
                    else:
                        tuned_sensitivity = self._sensitivity

                    # 3. Add to rolling window history
                    self._score_history.append(score)

                    # 4. Compute metrics
                    max_score = max(self._score_history) if self._score_history else 0.0
                    avg_score = sum(self._score_history) / len(self._score_history) if self._score_history else 0.0

                    # 5. Check consecutive detections
                    if len(self._score_history) >= self._min_consecutive_detections:
                        last_k_scores = list(self._score_history)[-self._min_consecutive_detections:]
                        consecutive_triggered = all(s >= self._lower_threshold for s in last_k_scores)
                    else:
                        consecutive_triggered = False

                    single_triggered = (score >= tuned_sensitivity)
                    is_triggered = single_triggered or consecutive_triggered

                    # 6. Check cooldown state
                    current_time = time.time()
                    in_cooldown = (current_time - self._last_trigger_time) < self._cooldown_duration

                    # 7. Detailed debug logging
                    log_throttled = (current_time - last_log_time >= 0.5)
                    if log_throttled or is_triggered:
                        if log_throttled:
                            last_log_time = current_time

                        min_amp = int(np.min(audio_chunk)) if len(audio_chunk) > 0 else 0
                        max_amp = int(np.max(audio_chunk)) if len(audio_chunk) > 0 else 0
                        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))) if len(audio_chunk) > 0 else 0.0
                        scores_str = ", ".join(f"{k}: {v:.3f}" for k, v in prediction.items())

                        logger.info(
                            "WW Debug - Score: %.3f | Roll Avg: %.3f | Roll Max: %.3f | "
                            "Tuned Sens: %.3f | Consecutive Trigger: %s | Single Trigger: %s | "
                            "Triggered: %s | In Cooldown: %s | RMS: %.2f | Scores: {%s}",
                            score,
                            avg_score,
                            max_score,
                            tuned_sensitivity,
                            consecutive_triggered,
                            single_triggered,
                            is_triggered,
                            in_cooldown,
                            rms,
                            scores_str,
                        )

                    if is_triggered and not in_cooldown:
                        logger.info(
                            "Wake word detected! Trigger source: %s (Raw Score: %.3f, Roll Avg: %.3f, Tuned Sensitivity: %.3f)",
                            "Single Frame" if single_triggered else "Consecutive Frames",
                            score,
                            avg_score,
                            tuned_sensitivity,
                        )
                        self._last_trigger_time = current_time
                        self._score_history.clear()
                        self._model.reset()
                        return True

            finally:
                if own_stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                    if audio:
                        try:
                            audio.terminate()
                        except Exception:
                            pass

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

    def listen_for_wake_word(self, check_running: Any = None, stream: Any = None) -> bool:
        """Simulates wake word listening and returns trigger boolean."""
        if check_running and not check_running():
            return False
        if not self._sequence:
            return False

        idx = min(self._call_count, len(self._sequence) - 1)
        triggered = self._sequence[idx]
        self._call_count += 1
        return triggered

