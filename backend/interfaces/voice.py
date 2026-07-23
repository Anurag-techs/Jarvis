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

import logging
import time

from backend.core.orchestrator import SystemOrchestrator
from backend.voice.base import BaseWakeWordDetector, SpeechToTextProvider
from backend.voice.service import TTSService
from backend.voice.wake_word import MockWakeWordDetector

logger = logging.getLogger("jarvis.interfaces.voice")


class VoiceController:
    """Hands-free continuous voice interface controller with stabilization and self-triggering prevention."""

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
    ) -> None:
        """Initialize VoiceController via dependency injection.

        Args:
            orchestrator: Injected SystemOrchestrator control plane instance.
            stt_provider: Injected SpeechToTextProvider implementation.
            wake_word_detector: Optional BaseWakeWordDetector implementation.
            tts_service: Optional TTSService for voice output.
            listen_timeout: Duration in seconds for active speech recording.
            post_response_timeout: Duration in seconds for follow-up listening window.
            cooldown_seconds: Duration in seconds to pause post-TTS before enabling STT.
            max_cycles: Optional outer-loop iteration cap (used for testing).
            max_conversation_turns: Optional inner conversation-loop turn cap (used for testing).
        """
        self._orchestrator = orchestrator
        self._stt = stt_provider
        self._wake_word = wake_word_detector or MockWakeWordDetector()
        self._tts = tts_service
        self._listen_timeout = listen_timeout
        self._post_response_timeout = post_response_timeout
        self._cooldown_seconds = cooldown_seconds
        self._max_cycles = max_cycles
        self._max_conversation_turns = max_conversation_turns

    def start(self) -> None:
        """Starts the hands-free continuous voice state machine."""
        print("\n====================================================")
        print("  JARVIS Hands-Free Voice Mode Online")
        print("====================================================\n")

        cycle_count = 0

        while True:
            if self._max_cycles is not None and cycle_count >= self._max_cycles:
                logger.info("Reached maximum execution cycles (%d). Exiting VoiceController.", self._max_cycles)
                break

            try:
                # --------------------------------------------------
                # State 1: Entering Standby (Listen for Wake Word)
                # --------------------------------------------------
                logger.info("Entering Standby")
                print("\n[Standby - Listening for 'Hey Jarvis'...]")

                try:
                    triggered = self._wake_word.listen_for_wake_word()
                except Exception as exc:
                    logger.error("Wake word detection error: %s", exc)
                    print(f"\n[Wake Word Error]: {exc}")
                    time.sleep(1)
                    cycle_count += 1
                    continue

                if not triggered:
                    cycle_count += 1
                    continue

                # --------------------------------------------------
                # State 2: Wake Word Triggered & Active Conversation Loop
                # --------------------------------------------------
                logger.info("Wake word detected")
                print("\n[Wake word detected!]")

                # Continuous conversation loop (Active Speech ➔ Processing ➔ TTS ➔ Cooldown ➔ Follow-up)
                self._run_conversation_loop()
                cycle_count += 1

            except (KeyboardInterrupt, EOFError):
                print("\n\nSession terminated by user. Goodbye!")
                break
            except Exception as err:
                logger.error("Unexpected error in VoiceController cycle: %s", err)
                print(f"\n[System Error]: {err}")
                cycle_count += 1
                continue

    def _run_conversation_loop(self) -> None:
        """Executes active conversation loop including post-TTS cooldown and follow-up window until silence."""
        in_followup = False
        turn_count = 0

        while True:
            if self._max_conversation_turns is not None and turn_count >= self._max_conversation_turns:
                logger.info("Reached maximum conversation turns (%d). Returning to Standby.", self._max_conversation_turns)
                break
            duration = self._post_response_timeout if in_followup else self._listen_timeout
            prompt_label = "[Follow-up listening (5s)...]" if in_followup else "[Listening...]"

            # --------------------------------------------------
            # State 3: Listening for Speech
            # --------------------------------------------------
            logger.info("Listening")
            print(f"\n{prompt_label}")

            # Capture microphone audio & transcribe (STT disabled during TTS)
            try:
                transcript = self._stt.listen_and_transcribe(duration=duration)
            except Exception as exc:
                logger.error("Microphone STT error during active listening: %s", exc)
                print(f"\n[Microphone Error]: {exc}")
                logger.info("Returning to Standby")
                print("\n[Returning to Standby]")
                break

            clean_transcript = transcript.strip() if transcript else ""

            # --------------------------------------------------
            # Handle empty or silent speech (Immediate Exit to Standby)
            # --------------------------------------------------
            if not clean_transcript:
                logger.info("Speech recognized: (empty)")
                if not in_followup:
                    fallback_msg = "I didn't catch that."
                    print(f"\nJarvis:\n{fallback_msg}")
                    if self._tts:
                        try:
                            logger.info("Speaking")
                            self._tts.speak(fallback_msg)
                            self._apply_cooldown()
                        except Exception as tts_err:
                            logger.warning("TTS speech failed: %s", tts_err)

                logger.info("Returning to Standby")
                print("\n[Returning to Standby]")
                break

            # --------------------------------------------------
            # State 4: Speech Recognized & Command Processing
            # --------------------------------------------------
            logger.info("Speech recognized: '%s'", clean_transcript)
            print(f"\nRecognized:\n{clean_transcript}")

            response = self._orchestrator.process(clean_transcript)
            print(f"\nJarvis:\n{response.text}")

            # --------------------------------------------------
            # State 5: Speaking Response & Microphone Cooldown
            # --------------------------------------------------
            if response.should_speak and self._tts:
                logger.info("Speaking")
                try:
                    self._tts.speak(response.text)
                except Exception as tts_err:
                    logger.warning("TTS speech execution failed: %s", tts_err)

            # Prevent self-triggering via post-TTS microphone cooldown delay
            self._apply_cooldown()

            # Transition into continuous follow-up window
            in_followup = True
            turn_count += 1

    def _apply_cooldown(self) -> None:
        """Internal helper applying post-TTS microphone cooldown pause to prevent feedback loops."""
        if self._cooldown_seconds > 0:
            logger.info("Cooldown")
            time.sleep(self._cooldown_seconds)
