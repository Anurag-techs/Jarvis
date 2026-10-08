"""
Integration and unit tests for JARVIS Voice Runtime Integration.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from backend.config.settings import Settings
from backend.core.runtime import RuntimeManager, RuntimeType
from backend.interfaces.voice import VoiceController, VoiceState
from backend.core.orchestrator import SystemOrchestrator
from backend.tools.registry import ToolRegistry
from backend.tools.base import BaseTool, ToolResult
from backend.voice.providers import MockSTTProvider, MockTTSProvider
from backend.voice.service import TTSService
from backend.voice.wake_word import MockWakeWordDetector


class TestVoiceRuntimeIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            debug=True,
            voice_mode=True,
            voice_provider="mock",
            stt_provider="mock",
            wakeword_provider="mock"
        )
        self.registry = ToolRegistry()
        self.llm = MagicMock()
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry
        )
        self.stt = MockSTTProvider(default_transcript="open chrome")
        self.mock_tts = MockTTSProvider()
        self.tts = TTSService(provider=self.mock_tts)

    @patch("backend.core.runtime.RuntimeManager.check_python_version")
    @patch("backend.core.runtime.RuntimeManager.check_dependencies")
    @patch("backend.core.runtime.RuntimeManager.check_microphone")
    def test_voice_runtime_selected_when_dependencies_exist(self, mock_mic, mock_deps, mock_py_ver) -> None:
        """Verify Voice runtime is selected when dependencies and mic exist."""
        mock_py_ver.return_value = True
        mock_deps.return_value = {
            "pyttsx3": True,
            "faster_whisper": True,
            "numpy": True,
            "openwakeword": True,
            "pyaudio": True
        }
        mock_mic.return_value = True

        manager = RuntimeManager(self.settings)
        selected, status = manager.select_runtime()
        self.assertEqual(selected, RuntimeType.VOICE)
        self.assertTrue(status["voice_supported"])

    @patch("backend.core.runtime.RuntimeManager.check_dependencies")
    @patch("backend.core.runtime.RuntimeManager.check_microphone")
    def test_console_runtime_selected_when_dependencies_missing(self, mock_mic, mock_deps) -> None:
        """Verify Console runtime is selected if some dependencies are missing."""
        mock_deps.return_value = {
            "pyttsx3": True,
            "faster_whisper": False,  # Missing STT library
            "numpy": True,
            "openwakeword": True,
            "pyaudio": True
        }
        mock_mic.return_value = True

        settings_prod = Settings(debug=False)
        manager = RuntimeManager(settings_prod)
        selected, status = manager.select_runtime()
        self.assertEqual(selected, RuntimeType.CONSOLE)
        self.assertFalse(status["voice_supported"])

    @patch("backend.core.runtime.RuntimeManager.check_python_version")
    @patch("backend.core.runtime.RuntimeManager.check_dependencies")
    @patch("backend.core.runtime.RuntimeManager.check_microphone")
    def test_runtime_selection_is_deterministic(self, mock_mic, mock_deps, mock_py_ver) -> None:
        """Verify subsequent selection calls remain identical."""
        mock_py_ver.return_value = True
        mock_deps.return_value = {
            "pyttsx3": True,
            "faster_whisper": True,
            "numpy": True,
            "openwakeword": True,
            "pyaudio": True
        }
        mock_mic.return_value = True

        manager = RuntimeManager(self.settings)
        sel1, _ = manager.select_runtime()
        sel2, _ = manager.select_runtime()
        self.assertEqual(sel1, sel2)

    def test_wake_word_triggers_listening_state(self) -> None:
        """Verify wake word detection transitions VoiceController to active listening."""
        states_reached = []

        def state_callback(old, new):
            states_reached.append(new)

        wake_word = MockWakeWordDetector()
        # Mock listen_for_wake_word to return True, then False to exit
        call_count = 0
        def listen_mock(check_running=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True
            controller.shutdown()
            return False

        wake_word.listen_for_wake_word = listen_mock

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            wake_word_detector=wake_word,
            tts_service=self.tts,
            max_cycles=2,
            on_state_change=state_callback
        )

        with patch("builtins.print"):
            controller.start()

        self.assertIn(VoiceState.RECORDING, states_reached)

    def test_voice_command_launches_desktop_automation(self) -> None:
        """Verify recognized voice command correctly runs registered desktop tools."""
        class MockDesktopTool(BaseTool):
            @property
            def name(self) -> str:
                return "desktop_automation"
            @property
            def description(self) -> str:
                return "Mock"
            def can_handle(self, query: str) -> bool:
                return "chrome" in query
            def execute(self, **kwargs) -> ToolResult:
                self.called_args = kwargs
                return ToolResult(success=True, message="Opening Chrome.")

        mock_tool = MockDesktopTool()
        self.registry.register(mock_tool)

        # Mock LLM to return tool call response
        from backend.core.models import ToolCall, AssistantResponse
        mock_response = AssistantResponse(
            success=True,
            text="Opening Chrome.",
            tool_calls=[ToolCall(tool="desktop_automation", arguments={"action": "open_browser", "args": {"url": "chrome"}})],
            should_speak=True
        )
        self.llm.generate_completion.return_value = mock_response

        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            tts_service=self.tts,
            max_cycles=1,
            max_conversation_turns=1
        )

        with patch("builtins.print"):
            with patch("builtins.input", side_effect=["", "exit"]):
                controller.start()

        self.assertIn("Opening Chrome.", self.mock_tts.spoken_messages)

    def test_voice_controller_shutdown_cleans_threads(self) -> None:
        """Verify shutdown method stops TTS service and cancels FSM loops."""
        mock_tts_service = MagicMock()
        controller = VoiceController(
            orchestrator=self.orchestrator,
            stt_provider=self.stt,
            tts_service=mock_tts_service
        )
        controller.shutdown()
        self.assertFalse(controller._running)
        mock_tts_service.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
