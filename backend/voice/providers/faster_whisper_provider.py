"""
JARVIS Faster-Whisper STT Provider.

1. Why this module exists:
   Integrates offline Speech-To-Text transcription using `faster-whisper` and microphone capture.

2. How it fits into the architecture:
   Concrete subclass of `SpeechToTextProvider`. Injected into voice controllers or startup container.

3. Which future modules will interact with it:
   - `backend.voice.controller.VoiceController`

4. Common mistakes to avoid:
   - Re-instantiating `WhisperModel` on every transcription call (model load takes 2-5 seconds).
   - Exposing PyAudio / C-extension exceptions directly to callers.

5. Possible future improvements:
   - Streaming voice activity detection (VAD) buffer trimming.
"""

import io
import logging
import os
import tempfile
import wave

from backend.core.exceptions import VoiceError
from backend.voice.base import SpeechToTextProvider

logger = logging.getLogger("jarvis.voice.providers.faster_whisper")


class FasterWhisperProvider(SpeechToTextProvider):
    """Offline STT Provider utilizing faster-whisper and default microphone audio capture."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """Initializes the WhisperModel instance once.

        Args:
            model_size: Model size string ('tiny', 'base', 'small', 'medium', 'large').
            device: Compute device ('cpu' or 'cuda').
            compute_type: Quantization type ('int8', 'float16', 'float32').

        Raises:
            VoiceError: If faster-whisper model fails to initialize.
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self.last_transcription_confidence = 1.0

        try:
            from faster_whisper import WhisperModel  # Deferred import for clean SDK loading

            # Initialize WhisperModel once and reuse across requests
            self._model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
            )
            logger.info("FasterWhisperProvider model initialized (%s, device=%s)", model_size, device)
        except Exception as exc:
            raise VoiceError(
                message=f"Failed to initialize faster-whisper model ({model_size}): {exc}",
                details={"provider": "faster_whisper", "model": model_size},
            ) from exc

    def listen_and_transcribe(self, duration: float = 5.0) -> str:
        """Captures audio from default microphone for specified duration and returns transcribed text.

        Args:
            duration: Audio recording duration in seconds.

        Returns:
            Transcribed text string.

        Raises:
            VoiceError: If microphone audio capture or transcription fails.
        """
        logger.info("Capturing microphone audio for %.1f seconds...", duration)
        try:
            import pyaudio

            # Audio capture parameters (16kHz mono 16-bit PCM for Whisper)
            sample_format = pyaudio.paInt16
            channels = 1
            rate = 16000
            chunk = 1024

            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=sample_format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )

            frames = []
            num_chunks = int(rate / chunk * duration)
            for _ in range(num_chunks):
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)
                except IOError as io_err:
                    logger.warning("PyAudio buffer overflow during STT capture (recovered): %s", io_err)
                    continue

            stream.stop_stream()
            stream.close()
            audio.terminate()

            # Save frames to temporary WAV buffer in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(audio.get_sample_size(sample_format))
                wf.setframerate(rate)
                wf.writeframes(b"".join(frames))

            wav_bytes = wav_buffer.getvalue()
            return self.transcribe_audio_bytes(wav_bytes)

        except Exception as exc:
            logger.error("Microphone audio capture failed: %s", exc)
            raise VoiceError(
                message=f"Microphone audio capture failed: {exc}",
                details={"provider": "faster_whisper", "duration": duration},
            ) from exc

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        """Transcribes raw WAV / PCM byte stream into text string.

        Args:
            audio_bytes: Byte payload of WAV file.

        Returns:
            Clean transcribed text string.

        Raises:
            VoiceError: If transcription processing fails.
        """
        if not audio_bytes:
            return ""

        tmp_file_path = None
        try:
            # Create a temporary file with delete=False on Windows.
            # We must close the file handle before transcribing so that PyAV/Faster Whisper
            # can access it without hitting a Windows PermissionError (sharing violation).
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file_path = tmp_file.name
            try:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
            finally:
                tmp_file.close()

            segments, info = self._model.transcribe(
                tmp_file_path,
                beam_size=5,
                patience=1.0,
                temperature=[0.0, 0.2, 0.4],
                vad_filter=True,
            )
            transcript_parts = []
            segment_probs = []
            import math
            for segment in segments:
                transcript_parts.append(segment.text.strip())
                prob = math.exp(segment.avg_logprob) if hasattr(segment, "avg_logprob") else 1.0
                segment_probs.append(prob)

            full_transcript = " ".join(transcript_parts).strip()
            self.last_transcription_confidence = sum(segment_probs) / len(segment_probs) if segment_probs else 1.0

            logger.info("FasterWhisper transcribed text: '%s' | Confidence: %.2f", full_transcript, self.last_transcription_confidence)
            return full_transcript
        except Exception as exc:
            raise VoiceError(
                message=f"FasterWhisper transcription failed: {exc}",
                details={"provider": "faster_whisper"},
            ) from exc
        finally:
            if tmp_file_path:
                try:
                    os.remove(tmp_file_path)
                except OSError:
                    # Ignore cleanup errors if the file is already gone or locked
                    pass
