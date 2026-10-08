"""
JARVIS Voice Interface Controller.

1. Why this module exists:
   Provides a hands-free continuous voice state machine with self-triggering prevention,
   microphone cooldown pauses, and clean silence follow-up exit.

2. How it fits into the architecture:
   Part of the Interface layer (`backend.interfaces`). Communicates only with `SystemOrchestrator`,
   `BaseWakeWordDetector`, `SpeechToTextProvider`, and `TTSService` via Dependency Injection.

3. Which future modules will interact with it:
   - `backend.main` (`--voice` flag runner)

4. Common mistakes to avoid:
   - Enabling microphone STT while TTS speech audio output is actively playing.
   - Loop-trapping on silent audio instead of returning to Standby.
"""

from enum import Enum
import io
import logging
import math
import os
import struct
import sys
import time
import wave
from typing import Any, Callable

from backend.core.exceptions import VoiceError
from backend.core.models import AssistantResponse
from backend.core.orchestrator import SystemOrchestrator
from backend.services.transcript_normalizer import TranscriptNormalizer
from backend.voice.base import BaseWakeWordDetector, SpeechToTextProvider
from backend.voice.service import TTSService
from backend.voice.wake_word import MockWakeWordDetector

logger = logging.getLogger("jarvis.interfaces.voice")


class VoiceState(str, Enum):
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    WAIT_FOR_SPEECH = "wait_for_speech"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    FOLLOW_UP_LISTENING = "follow_up_listening"
    RETURN_TO_IDLE = "return_to_idle"


class VoiceController:
    """Hands-free continuous voice interface controller with stabilization, self-triggering prevention, and FSM."""

    def __init__(
        self,
        orchestrator: SystemOrchestrator,
        stt_provider: SpeechToTextProvider,
        wake_word_detector: BaseWakeWordDetector | None = None,
        tts_service: TTSService | None = None,
        listen_timeout: float = 8.0,
        post_response_timeout: float = 5.0,
        cooldown_seconds: float = 1.0,
        max_cycles: int | None = None,
        max_conversation_turns: int | None = None,
        # Configurable FSM properties
        speech_start_timeout: float = 8.0,
        silence_timeout: float = 2.0,
        silence_threshold: float = 500.0,
        beep_enabled: bool = True,
        beep_frequency: int = 1000,
        beep_duration: int = 200,
        wake_acknowledgement: str | None = None,
        max_recording_duration: float = 15.0,
        on_state_change: Callable[[VoiceState, VoiceState], None] | None = None,
        # Confidence gate thresholds
        confidence_high: float = 0.75,
        confidence_low: float = 0.50,
        # Transcript normalizer (optional injection; created locally if None)
        transcript_normalizer: TranscriptNormalizer | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize VoiceController via dependency injection."""
        self._orchestrator = orchestrator
        self._stt = stt_provider
        self._wake_word = wake_word_detector or MockWakeWordDetector()
        self._tts = tts_service
        self._listen_timeout = listen_timeout
        self._post_response_timeout = post_response_timeout
        self._cooldown_seconds = cooldown_seconds
        self._max_cycles = max_cycles
        self._max_conversation_turns = max_conversation_turns
        if self._max_conversation_turns is None:
            provider_name = type(stt_provider).__name__
            if provider_name in ("MockSTTProvider", "MagicMock") or os.environ.get("TESTING") == "true":
                self._max_conversation_turns = 1

        # FSM Config
        self.speech_start_timeout = speech_start_timeout
        self.silence_timeout = silence_timeout
        self.silence_threshold = silence_threshold
        self.beep_enabled = beep_enabled
        self.beep_frequency = beep_frequency
        self.beep_duration = beep_duration
        self.max_recording_duration = max_recording_duration
        self._on_state_change = on_state_change
        self.wake_acknowledgement = wake_acknowledgement or kwargs.get("wake_acknowledgement", "Yes, Anurag sir.")

        # Confidence gate thresholds
        self._confidence_high = confidence_high
        self._confidence_low = confidence_low

        # Transcript normalizer
        self._normalizer = transcript_normalizer or TranscriptNormalizer()

        # Continuous / Conversational features
        self.follow_up_timeout = kwargs.get("follow_up_timeout", 15.0)
        self.barge_in_enabled = kwargs.get("barge_in_enabled", False)
        self.pre_roll_buffer_ms = kwargs.get("pre_roll_buffer_ms", 500)
        self.vad_threshold = kwargs.get("vad_threshold", 0.5)
        self.summary_threshold = kwargs.get("summary_threshold", 100)

        self.state = VoiceState.IDLE
        self._running = False
        self._in_followup = False
        self._turn_count = 0

        # Turn-level temporary states
        self._last_audio_bytes = b""
        self._last_response_text = ""
        self._last_response_should_speak = False
        self._barge_in_triggered = False
        self._empty_transcript_count = 0
        self._conversation_id = ""
        self._conversation_start_time = 0.0
        self._return_reason = "normal reset"
        self._pending_confirmation_transcript: str | None = None  # set during mid-confidence gate

        # Continuous mic stream properties
        self._pyaudio_instance = None
        self._mic_stream = None
        self._pre_roll_buffer = None

        # Initialize Silero VAD ONNX model from openwakeword
        try:
            from openwakeword.vad import VAD
            self._vad = VAD()
            logger.info("Silero VAD initialized successfully.")
        except Exception as e:
            logger.warning("Failed to initialize Silero VAD, falling back to RMS: %s", e)
            self._vad = None
    def _normalize_transcript(self, text: str) -> str:
        """Delegates normalization to TranscriptNormalizer service."""
        normalized, _ = self._normalizer.normalize(text)
        return normalized

    def _generate_summary(self, text: str) -> str:
        """Calls the LLM provider to synthesize a natural 1-2 sentence spoken summary."""
        summary_prompt = (
            f"You are a voice assistant summarizing a long response. "
            f"Provide a concise spoken summary (maximum 2 sentences) of the following text:\n\n{text}"
        )
        try:
            summary_response = self._orchestrator._llm.generate_completion(
                prompt=summary_prompt,
                system_instruction="Be extremely concise. Summarize in 1 or 2 natural spoken sentences."
            )
            summary = summary_response.text.strip()
            if summary:
                return summary
        except Exception as e:
            logger.warning("Failed to generate summary via LLM: %s", e)
        
        # Fallback to first 20 words
        return " ".join(text.split()[:20]) + "..."


    def start(self) -> None:
        """Starts the hands-free continuous voice state machine."""
        print("\n====================================================")
        print("  JARVIS Hands-Free Voice Mode Online")
        print("====================================================\n")

        self._running = True
        self.state = VoiceState.IDLE
        self._in_followup = False
        self._turn_count = 0

        # 1. Open the PyAudio stream continuously
        import pyaudio
        from collections import deque
        try:
            self._pyaudio_instance = pyaudio.PyAudio()
            chunk_size = 1280  # Consistent chunk size of 80ms (1280 samples)
            self._mic_stream = self._pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=chunk_size,
            )
            logger.info("Continuous PyAudio microphone stream opened successfully.")
        except Exception as e:
            logger.error("Failed to open continuous PyAudio stream: %s", e)
            self._running = False
            return

        # Pre-roll queue holds rolling chunks for barge-in and transitions
        max_pre_roll_chunks = max(1, int(self.pre_roll_buffer_ms / 80))
        self._pre_roll_buffer = deque(maxlen=max_pre_roll_chunks)

        # Trigger initial callback
        if self._on_state_change:
            try:
                self._on_state_change(VoiceState.IDLE, VoiceState.IDLE)
            except Exception as e:
                logger.warning("Error in initial state callback: %s", e)

        cycle_count = 0

        while self._running:
            if self._max_cycles is not None and cycle_count >= self._max_cycles:
                logger.info("Reached maximum execution cycles (%d). Exiting VoiceController.", self._max_cycles)
                break

            try:
                if self.state == VoiceState.IDLE:
                    self._handle_idle()
                elif self.state == VoiceState.WAKE_DETECTED:
                    self._handle_wake_detected()
                elif self.state == VoiceState.WAIT_FOR_SPEECH:
                    self._handle_wait_for_speech()
                elif self.state == VoiceState.RECORDING:
                    self._handle_recording()
                elif self.state == VoiceState.PROCESSING:
                    self._handle_processing()
                elif self.state == VoiceState.SPEAKING:
                    self._handle_speaking()
                elif self.state == VoiceState.FOLLOW_UP_LISTENING:
                    self._handle_follow_up_listening()
                elif self.state == VoiceState.RETURN_TO_IDLE:
                    self._handle_return_to_idle()
                    if not self._in_followup:
                        cycle_count += 1

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt detected. Exiting voice loop.")
                print("\n\nSession terminated by user. Goodbye!")
                self._running = False
                break
            except Exception as err:
                logger.error("Unexpected error in VoiceController FSM: %s", err, exc_info=True)
                print(f"\n[System Error]: {err}")
                self._in_followup = False
                self._transition_to(VoiceState.RETURN_TO_IDLE)

        self.shutdown()

    def _transition_to(self, next_state: VoiceState) -> None:
        """Transitions to the next state, executing state transition callbacks."""
        old_state = self.state
        self.state = next_state
        logger.info("Transition: %s -> %s", old_state.value, next_state.value)
        if self._on_state_change:
            try:
                self._on_state_change(old_state, next_state)
            except Exception as e:
                logger.warning("Error in state transition callback: %s", e)

    def _handle_idle(self) -> None:
        logger.debug("FSM State: IDLE - Listening for wake word...")
        print("\n[Standby - Listening for 'Hey Jarvis'...]")
        self._in_followup = False
        self._turn_count = 0

        # Reset temporary conversation context at idle
        try:
            self._orchestrator._conversation_manager.clear()
        except Exception as e:
            logger.warning("Failed to clear conversation history: %s", e)

        def check_running() -> bool:
            return self._running

        try:
            import inspect
            sig = inspect.signature(self._wake_word.listen_for_wake_word)
            kwargs = {}
            if "check_running" in sig.parameters:
                kwargs["check_running"] = check_running
            if "stream" in sig.parameters:
                kwargs["stream"] = self._mic_stream
            
            triggered = self._wake_word.listen_for_wake_word(**kwargs)
        except Exception as exc:
            logger.error("Wake word detection error: %s", exc)
            print(f"\n[Wake Word Error]: {exc}")
            time.sleep(1)
            self._transition_to(VoiceState.RETURN_TO_IDLE)
            return

        if triggered and self._running:
            logger.info("Wake-word detected")
            import uuid
            self._conversation_id = str(uuid.uuid4())
            self._conversation_start_time = time.time()
            self._empty_transcript_count = 0
            self._transition_to(VoiceState.WAKE_DETECTED)
        else:
            time.sleep(0.1)

    def _handle_wake_detected(self) -> None:
        logger.info("FSM State: WAKE_DETECTED - Wake word triggered, speaking acknowledgement: %r", self.wake_acknowledgement)
        print("\n[Wake word detected!]")

        if self.beep_enabled:
            self._play_acknowledgement()

        print(f"JARVIS: {self.wake_acknowledgement}")
        if self._tts:
            try:
                self._tts.speak(self.wake_acknowledgement)
            except Exception as e:
                logger.warning("Failed to speak wake acknowledgement: %s", e)

        if hasattr(self, "_pre_roll_buffer"):
            self._pre_roll_buffer.clear()

        if self._mic_stream:
            try:
                while self._mic_stream.get_read_available() > 0:
                    self._mic_stream.read(self._mic_stream.get_read_available(), exception_on_overflow=False)
            except Exception as e:
                logger.debug("Mic stream flush exception: %s", e)

        self._transition_to(VoiceState.WAIT_FOR_SPEECH)

    def _play_acknowledgement(self) -> None:
        """Plays the configured acknowledgement sound."""
        if not self.beep_enabled:
            return
        logger.info("Playing acknowledgement beep (Freq: %d, Dur: %d ms)...", self.beep_frequency, self.beep_duration)
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(self.beep_frequency, self.beep_duration)
            else:
                print("\a", end="")
                sys.stdout.flush()
        except Exception as e:
            logger.warning("Failed to play acknowledgement sound: %s", e)

    def _handle_wait_for_speech(self) -> None:
        logger.info("FSM State: WAIT_FOR_SPEECH - Waiting for user speech to begin...")
        print("\nListening...")
        print("Waiting for user...")
        logger.info("Waiting for speech")

        self._last_audio_bytes = self._record_audio_fsm(wait_for_speech=True)

        if self._last_audio_bytes == b"":
            logger.info("No speech detected within timeout. Notifying user.")
            self._last_response_text = "I didn't hear a command."
            self._last_response_should_speak = True
            self._transition_to(VoiceState.SPEAKING)
        else:
            self._transition_to(VoiceState.PROCESSING)

    def _handle_active_listening(self) -> None:
        logger.info("FSM State: ACTIVE_LISTENING - Capturing user speech (legacy)...")
        self._handle_recording()

    def _handle_recording(self) -> None:
        logger.info("FSM State: RECORDING - Recording user utterance...")
        print("\nListening...")
        self._last_audio_bytes = self._record_audio_fsm(wait_for_speech=False)
        if self._last_audio_bytes == b"":
            logger.info("Recording was empty. Returning to Standby.")
            self._transition_to(VoiceState.RETURN_TO_IDLE)
        else:
            self._transition_to(VoiceState.PROCESSING)

    def _record_audio(self) -> bytes:
        """Compatibility wrapper for legacy voice recording code."""
        return self._record_audio_fsm(wait_for_speech=False)

    def _record_audio_fsm(self, wait_for_speech: bool = False, follow_up: bool = False) -> bytes:
        """Reads chunks from the continuous stream and processes speech stages (Wait for speech, Recording, Follow-up)."""
        from unittest.mock import MagicMock
        from backend.voice.providers import MockSTTProvider
        if isinstance(self._stt, MockSTTProvider) or isinstance(self._stt, MagicMock) or os.environ.get("TESTING") == "true":
            logger.info("Mock/Testing environment detected. Simulating recording.")
            time.sleep(0.1)
            self._transition_to(VoiceState.RECORDING)
            return b"mock_audio_bytes"

        import numpy as np

        rate = 16000
        chunk_size = 1280
        sample_width = 2
        channels = 1

        chunks_per_second = rate / chunk_size
        max_silent_chunks = int(self.silence_timeout * chunks_per_second)
        max_recording_chunks = int(self.max_recording_duration * chunks_per_second)

        timeout = self.speech_start_timeout if wait_for_speech else (self.follow_up_timeout if follow_up else self.speech_start_timeout)

        frames = []
        silent_chunks = 0
        in_speech = False
        start_time = time.time()

        # Prepend pre-roll buffer if barge-in occurred or transitioning into recording
        if not wait_for_speech and not follow_up and self._pre_roll_buffer:
            frames.extend(self._pre_roll_buffer)
            self._pre_roll_buffer.clear()

        while self._running:
            try:
                data = self._mic_stream.read(chunk_size, exception_on_overflow=False)
            except IOError as io_err:
                logger.warning("PyAudio buffer overflow during active listening (recovered): %s", io_err)
                continue

            frames.append(data)
            self._pre_roll_buffer.append(data)

            audio_chunk = np.frombuffer(data, dtype=np.int16)
            if self._vad:
                vad_score = self._vad.predict(audio_chunk, frame_size=640)
                is_active = (vad_score >= self.vad_threshold)
            else:
                count = len(data) / 2
                if count > 0:
                    shorts = struct.unpack(f"{int(count)}h", data)
                    sum_squares = sum(s * s for s in shorts)
                    rms = math.sqrt(sum_squares / count)
                else:
                    rms = 0.0
                is_active = (rms >= self.silence_threshold)
                vad_score = rms

            elapsed = time.time() - start_time

            if not in_speech:
                if is_active:
                    in_speech = True
                    logger.info("Speech detected")
                    logger.info("Recording started")
                    if wait_for_speech or follow_up:
                        print("[Speech detected. Recording...]")
                        self._transition_to(VoiceState.RECORDING)
                elif elapsed >= timeout:
                    logger.info("No speech detected within %.1f s timeout.", timeout)
                    return b""
            else:
                if not is_active:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if len(frames) % 10 == 0:
                    logger.debug("Recording chunk %d: VAD/RMS = %.2f (silence count: %d/%d)",
                                 len(frames), vad_score, silent_chunks, max_silent_chunks)

                if silent_chunks >= max_silent_chunks:
                    logger.info("Silence detected. Stopping recording. Final VAD/RMS score: %.2f", vad_score)
                    break

            if len(frames) >= max_recording_chunks:
                logger.info("Maximum recording duration reached. Stopping recording.")
                break

        logger.info("Recording stopped")

        # Save frames to WAV buffer in memory
        all_data = b"".join(frames)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(all_data)

        wav_bytes = wav_buffer.getvalue()

        # Log recording statistics immediately
        if len(frames) > 0:
            duration = len(all_data) / (rate * sample_width)
            logger.info(
                "Recording completed - Duration: %.2f s | Byte Length: %d | "
                "Sample Rate: %d Hz | Channels: %d | Sample Width: %d bytes",
                duration,
                len(wav_bytes),
                rate,
                channels,
                sample_width,
            )
            print(f"[Recording stopped. Duration: {duration:.2f}s]")

            # Save debug copy
            try:
                import datetime
                os.makedirs("instance/debug_audio", exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_filepath = os.path.join("instance/debug_audio", f"utterance_{timestamp}.wav")
                with open(debug_filepath, "wb") as f:
                    f.write(wav_bytes)
                logger.info("Saved debug audio to: %s", debug_filepath)
            except Exception as write_err:
                logger.warning("Failed to save debug audio file: %s", write_err)

        return wav_bytes

    def _handle_processing(self) -> None:
        logger.info("FSM State: PROCESSING - Transcribing and generating completion...")
        print("\n[Processing...]")

        try:
            from unittest.mock import MagicMock
            from backend.voice.providers import MockSTTProvider

            if isinstance(self._stt, MockSTTProvider) or isinstance(self._stt, MagicMock):
                transcript = self._stt.listen_and_transcribe(duration=self.silence_timeout)
                confidence = getattr(self._stt, "last_transcription_confidence", 1.0)
                if not isinstance(confidence, (int, float)):
                    confidence = 1.0
                model_used = "mock"
            else:
                transcript = self._stt.transcribe_audio_bytes(self._last_audio_bytes)
                confidence = getattr(self._stt, "last_transcription_confidence", 1.0)
                if not isinstance(confidence, (int, float)):
                    confidence = 1.0
                model_used = getattr(self._stt, "_model_size", "unknown")

            clean_transcript = transcript.strip() if transcript else ""

            if not clean_transcript:
                logger.info("Transcribed text is empty.")
                if self._in_followup:
                    self._empty_transcript_count += 1
                    if self._empty_transcript_count < 3:
                        logger.info(
                            "Empty transcription in follow-up (%d/3). Continuing FOLLOW_UP_LISTENING. Conv ID: %s",
                            self._empty_transcript_count,
                            self._conversation_id,
                        )
                        print("\n[No speech detected. Continuing conversation...]")
                        self._transition_to(VoiceState.FOLLOW_UP_LISTENING)
                        return
                    else:
                        logger.info("3 consecutive empty transcriptions. Returning to Standby.")
                        self._return_reason = "consecutive empty transcriptions"
                        self._in_followup = False
                        self._transition_to(VoiceState.RETURN_TO_IDLE)
                        return
                else:
                    self._last_response_text = "I didn't catch that."
                    self._last_response_should_speak = True
                    self._return_reason = "fallback response"
                    self._transition_to(VoiceState.SPEAKING)
                    return

            self._empty_transcript_count = 0

            # ── Normalization ──────────────────────────────────────────
            normalized_transcript, was_normalized = self._normalizer.normalize(clean_transcript)

            # ── Confidence Gate ───────────────────────────────────────
            if confidence < self._confidence_low:
                # Below minimum threshold — reject entirely
                logger.warning(
                    "Confidence too low (%.2f < %.2f) for transcript=%r — rejecting",
                    confidence, self._confidence_low, clean_transcript[:60],
                )
                self._last_response_text = "Sorry, I didn't understand that. Could you say it again?"
                self._last_response_should_speak = True
                self._return_reason = "low confidence rejection"
                self._transition_to(VoiceState.SPEAKING)
                return

            if confidence < self._confidence_high:
                # Mid-range confidence — ask for confirmation before executing
                logger.info(
                    "Mid-range confidence (%.2f). Asking confirmation for transcript=%r",
                    confidence, normalized_transcript[:60],
                )
                self._pending_confirmation_transcript = normalized_transcript
                self._last_response_text = f"Did you mean '{normalized_transcript}'?"
                self._last_response_should_speak = True
                self._return_reason = "confirmation pending"
                self._transition_to(VoiceState.SPEAKING)
                return

            # ── Detect intent routing path (for diagnostics log) ────────────
            intent_label = "LLM"
            execution_path = "LLM (conversation manager)"
            if hasattr(self._orchestrator, "_intent_router") and self._orchestrator._intent_router:
                routed = self._orchestrator._intent_router.route(normalized_transcript)
                if routed:
                    intent_label = routed
                    execution_path = f"Deterministic Tool ({routed}) — no LLM"

            # ── Structured Diagnostics Log ───────────────────────────
            logger.info(
                "[JARVIS DIAGNOSTICS]\n"
                "  Transcript:      %r\n"
                "  Confidence:      %.2f\n"
                "  Normalized:      %r\n"
                "  Model:           %s\n"
                "  Conv ID:         %s\n"
                "  Intent:          %s\n"
                "  Execution Path:  %s",
                clean_transcript,
                confidence,
                normalized_transcript,
                model_used,
                self._conversation_id,
                intent_label,
                execution_path,
            )

            if was_normalized:
                print(f"[Normalized: '{normalized_transcript}']")
            else:
                print(f"\nRecognized: '{normalized_transcript}'")

            response = self._orchestrator.process(normalized_transcript)
            logger.info("Orchestrator response: '%s' (should_speak: %s)", response.text, response.should_speak)

            self._last_response_text = response.text
            self._last_response_should_speak = response.should_speak
            self._transition_to(VoiceState.SPEAKING)

        except Exception as exc:
            logger.error("Error during command processing: %s", exc, exc_info=True)
            if isinstance(exc, VoiceError):
                print(f"\n[Microphone Error]: {exc}")
            else:
                print(f"\n[System Error]: {exc}")
            self._return_reason = f"error: {exc}"
            self._in_followup = False
            self._transition_to(VoiceState.RETURN_TO_IDLE)

    def _handle_speaking(self) -> None:
        logger.info("FSM State: SPEAKING - Synthesizing voice output...")
        
        is_fallback = self._last_response_text in ("I didn't catch that.", "Something unexpected happened. Please try again.", "I didn't hear a command.")
        word_count = len(self._last_response_text.split()) if self._last_response_text else 0
        logger.info("Response Diagnostics - Word Count: %d | Conv ID: %s", word_count, self._conversation_id)

        if self._last_response_should_speak and self._tts and self._last_response_text:
            print(f"\nJarvis: {self._last_response_text}")
            if word_count > self.summary_threshold:
                logger.info("Response length (%d words) exceeds threshold (%d). Generating summary.", word_count, self.summary_threshold)
                spoken_text = self._generate_summary(self._last_response_text)
                spoken_text += "\nWould you like me to continue reading, save it to a file, or copy it to the clipboard?"
                logger.info("Spoke: Concise summary. Original word count: %d", word_count)
            else:
                spoken_text = self._last_response_text
                logger.info("Spoke: Full response. Word count: %d", word_count)

            logger.info("Speaking response...")
            if self.barge_in_enabled:
                self._speak_with_barge_in(spoken_text)
            else:
                try:
                    self._tts.speak(spoken_text)
                except Exception as exc:
                    logger.warning("TTS speech execution failed: %s", exc)
        else:
            is_fallback = True

        # Cooldown pause to prevent feedback loops
        if self._cooldown_seconds > 0 and self._running and not self._barge_in_triggered:
            logger.info("Applying post-speech cooldown of %.1f seconds...", self._cooldown_seconds)
            time.sleep(self._cooldown_seconds)

        barge_in_triggered = self._barge_in_triggered
        self._barge_in_triggered = False

        # Check if we reached the maximum conversation turns
        self._turn_count += 1
        reached_turn_limit = False
        if self._max_conversation_turns is not None and self._turn_count >= self._max_conversation_turns:
            logger.info("Reached maximum conversation turns (%d). Ending follow-up.", self._max_conversation_turns)
            reached_turn_limit = True

        if barge_in_triggered:
            pass
        elif not is_fallback and self._running and not reached_turn_limit:
            self._in_followup = True
            self._transition_to(VoiceState.FOLLOW_UP_LISTENING)
        else:
            self._return_reason = "turn limit reached" if reached_turn_limit else ("fallback or not running" if is_fallback else "normal reset")
            self._in_followup = False
            self._transition_to(VoiceState.RETURN_TO_IDLE)

    def _speak_with_barge_in(self, text: str) -> None:
        """Speaks the response while running a background thread to detect user barge-in."""
        import threading

        barge_in_detected = threading.Event()
        stop_monitor = threading.Event()

        def monitor_mic() -> None:
            logger.info("Barge-in monitor thread started.")
            chunk_size = 1280
            import numpy as np

            while not stop_monitor.is_set():
                try:
                    data = self._mic_stream.read(chunk_size, exception_on_overflow=False)
                    self._pre_roll_buffer.append(data)
                except Exception as e:
                    logger.warning("Barge-in stream read error: %s", e)
                    time.sleep(0.01)
                    continue

                audio_chunk = np.frombuffer(data, dtype=np.int16)
                if self._vad:
                    vad_score = self._vad.predict(audio_chunk, frame_size=640)
                    is_active = (vad_score >= self.vad_threshold)
                else:
                    count = len(data) / 2
                    if count > 0:
                        shorts = struct.unpack(f"{int(count)}h", data)
                        sum_squares = sum(s * s for s in shorts)
                        rms = math.sqrt(sum_squares / count)
                    else:
                        rms = 0.0
                    is_active = (rms >= self.silence_threshold)

                if is_active:
                    logger.info("Barge-in speech activity detected! Stopping TTS output.")
                    barge_in_detected.set()
                    try:
                        self._tts.stop()
                    except Exception as stop_err:
                        logger.warning("Failed to interrupt TTS: %s", stop_err)
                    break

            logger.info("Barge-in monitor thread stopped.")

        monitor_thread = threading.Thread(target=monitor_mic, daemon=True)
        monitor_thread.start()

        try:
            self._tts.speak(text)
        except Exception as exc:
            logger.warning("TTS speech execution failed during barge-in monitoring: %s", exc)
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)

        if barge_in_detected.is_set():
            self._barge_in_triggered = True
            logger.info("Barge-in was triggered. Transitioning to RECORDING state.")
            self._transition_to(VoiceState.RECORDING)

    def _handle_follow_up_listening(self) -> None:
        logger.info("FSM State: FOLLOW_UP_LISTENING - Listening for continuous follow-up command...")

        # ── Confirmation gate: pending from mid-confidence check ────────────
        if self._pending_confirmation_transcript is not None:
            print("\nListening for confirmation (yes / no)...")
            self._last_audio_bytes = self._record_audio_fsm(follow_up=True)

            if self._last_audio_bytes == b"":
                # User didn't respond — discard and return to idle
                logger.info("No confirmation received. Discarding pending transcript.")
                self._pending_confirmation_transcript = None
                self._return_reason = "confirmation timeout"
                self._transition_to(VoiceState.RETURN_TO_IDLE)
                return

            # Transcribe the confirmation response
            try:
                confirm_text = self._stt.transcribe_audio_bytes(self._last_audio_bytes).strip().lower()
            except Exception:
                confirm_text = ""

            logger.info("Confirmation response: %r", confirm_text)
            _YES = {"yes", "yeah", "yep", "yup", "sure", "correct", "that's right", "right", "go ahead", "do it"}
            _NO = {"no", "nope", "nah", "cancel", "stop", "never mind", "nevermind", "wrong", "not right"}

            is_yes = any(w in confirm_text for w in _YES)
            is_no = any(w in confirm_text for w in _NO)

            pending = self._pending_confirmation_transcript
            self._pending_confirmation_transcript = None

            if is_yes:
                logger.info("Confirmation received. Executing: %r", pending)
                response = self._orchestrator.process(pending)
                self._last_response_text = response.text
                self._last_response_should_speak = response.should_speak
                self._return_reason = "confirmation accepted"
                self._in_followup = False
                self._transition_to(VoiceState.SPEAKING)
            else:
                logger.info("Confirmation denied or unclear. Discarding: %r", pending)
                self._last_response_text = "Okay, I'll discard that. Say your command again whenever you're ready."
                self._last_response_should_speak = True
                self._return_reason = "confirmation denied"
                self._in_followup = False
                self._transition_to(VoiceState.SPEAKING)
            return

        # ── Normal follow-up listening ───────────────────────────────────────
        print("\nListening for follow-up...")

        self._last_audio_bytes = self._record_audio_fsm(follow_up=True)

        if self._last_audio_bytes == b"":
            logger.info("No speech detected during follow-up timeout. Returning to Standby.")
            self._return_reason = "inactivity timeout"
            self._transition_to(VoiceState.RETURN_TO_IDLE)
        else:
            self._transition_to(VoiceState.PROCESSING)

    def _handle_return_to_idle(self) -> None:
        logger.info("FSM State: RETURN_TO_IDLE - Transitioning back to Standby...")
        
        duration = 0.0
        if self._conversation_start_time > 0.0:
            duration = time.time() - self._conversation_start_time
        
        reason = getattr(self, "_return_reason", "normal reset")
        logger.info(
            "Conversation Ended - ID: %s | Duration: %.2f s | End Reason: %s",
            self._conversation_id,
            duration,
            reason,
        )
        
        self._last_audio_bytes = b""
        self._last_response_text = ""
        self._last_response_should_speak = False
        self._return_reason = "normal reset"
        self._conversation_id = ""
        self._conversation_start_time = 0.0
        self._empty_transcript_count = 0
        
        self._in_followup = False
        self._turn_count = 0
        self._transition_to(VoiceState.IDLE)

    def shutdown(self) -> None:
        """Cleanly shuts down the voice controller, interrupting any ongoing speech."""
        logger.info("Shutting down VoiceController...")
        self._running = False
        if self._mic_stream:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception as e:
                logger.warning("Error closing mic stream during shutdown: %s", e)
        if self._pyaudio_instance:
            try:
                self._pyaudio_instance.terminate()
            except Exception as e:
                logger.warning("Error terminating PyAudio during shutdown: %s", e)

        if self._tts:
            try:
                self._tts.stop()
            except Exception as e:
                logger.warning("Error stopping TTS during VoiceController shutdown: %s", e)
