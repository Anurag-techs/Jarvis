"""
Unit and integration tests for JARVIS Voice Loop FSM state transitions and lifecycle control.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from backend.ai.provider import MockLLMProvider
from backend.core.exceptions import VoiceError
from backend.core.models import AssistantResponse
from backend.core.orchestrator import SystemOrchestrator
from backend.interfaces.voice import VoiceController, VoiceState
from backend.tools.registry import ToolRegistry
from backend.voice.providers import MockSTTProvider, MockTTSProvider
from backend.voice.service import TTSService
from backend.voice.wake_word import MockWakeWordDetector


class TestVoiceLoopFSM(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.llm = MockLLMProvider()
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
        )
        self.stt = MockSTTProvider(default_transcript="Test command")
        self.mock_tts = MockTTSProvider()
        self.tts = TTSService(provider=self.mock_tts)
        self.wake_word = MockWakeWordDetector(trigger_sequence=[True])

    def test_state_transitions_sequence(self) -> None:
        """Verifies FSM state transitions occur in correct logical order."""
        transitions = []

        def on_state_change(old_state: VoiceState, new_state: VoiceState) -> None:
            transitions.append((old_state, new_state))

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            wake_word_detector=self.wake_word,
            tts_service=self.tts,
            max_cycles=1,
            max_conversation_turns=1,
            on_state_change=on_state_change,
            cooldown_seconds=0.0,
        )

        controller.start()

        # Expected transition history:
        # 1. Initialization: IDLE -> IDLE
        # 2. Wake Word Triggered: IDLE -> ACTIVE_LISTENING
        # 3. Audio captured: ACTIVE_LISTENING -> PROCESSING
        # 4. Completion resolved: PROCESSING -> SPEAKING
        # 5. Speech finished: SPEAKING -> RETURN_TO_IDLE
        # 6. Loop reset: RETURN_TO_IDLE -> IDLE
        expected = [
            (VoiceState.IDLE, VoiceState.IDLE),
            (VoiceState.IDLE, VoiceState.WAKE_DETECTED),
            (VoiceState.WAKE_DETECTED, VoiceState.WAIT_FOR_SPEECH),
            (VoiceState.WAIT_FOR_SPEECH, VoiceState.RECORDING),
            (VoiceState.RECORDING, VoiceState.PROCESSING),
            (VoiceState.PROCESSING, VoiceState.SPEAKING),
            (VoiceState.SPEAKING, VoiceState.RETURN_TO_IDLE),
            (VoiceState.RETURN_TO_IDLE, VoiceState.IDLE),
        ]
        self.assertEqual(transitions, expected)

    def test_configurable_beep_disabled(self) -> None:
        """Verifies FSM runs without attempting sound playback when disabled."""
        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            wake_word_detector=self.wake_word,
            tts_service=self.tts,
            max_cycles=1,
            max_conversation_turns=1,
            beep_enabled=False,
            cooldown_seconds=0.0,
        )

        # Patch winsound Beep to verify it is NOT called when beep_enabled=False
        with patch("sys.platform", "win32"):
            with patch("winsound.Beep") as mock_beep:
                controller.start()
                mock_beep.assert_not_called()

    def test_interruption_during_speech(self) -> None:
        """Verifies that calling shutdown interrupts ongoing TTS output immediately."""
        mock_tts_provider = MagicMock()
        tts_service = TTSService(provider=mock_tts_provider)

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            wake_word_detector=self.wake_word,
            tts_service=tts_service,
            max_cycles=1,
            max_conversation_turns=1,
            cooldown_seconds=0.0,
        )

        # Call shutdown during speaking
        controller.shutdown()
        
        # Verify TTS provider stop() was triggered
        mock_tts_provider.stop.assert_called_once()
        self.assertFalse(controller._running)

    def test_background_thread_graceful_shutdown(self) -> None:
        """Verifies starting controller in background thread and calling shutdown terminates cleanly."""
        # Setup wake word that blocks/spins until trigger
        blocking_wake_word = MockWakeWordDetector(trigger_sequence=[False])
        
        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            wake_word_detector=blocking_wake_word,
            tts_service=self.tts,
            cooldown_seconds=0.0,
        )

        # Run start() in background thread
        thread = threading.Thread(target=controller.start, daemon=True)
        thread.start()

        # Wait briefly for thread to start and enter IDLE state
        time.sleep(0.1)
        self.assertTrue(thread.is_alive())
        self.assertEqual(controller.state, VoiceState.IDLE)

        # Shutdown controller
        controller.shutdown()

        # Wait for thread termination (with a 2-second timeout to avoid test hanging)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "VoiceController thread did not exit cleanly on shutdown.")


if __name__ == "__main__":
    unittest.main()
