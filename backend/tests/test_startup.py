"""
Unit tests for JARVIS Sprint 1 StartupManager and JarvisApplication.
"""

import unittest
from unittest.mock import patch

from backend.core.exceptions import ConfigurationError
from backend.core.startup import JarvisApplication, StartupManager


class TestStartupManager(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = StartupManager()

    def test_bootstrap_success(self) -> None:
        """Verifies that StartupManager.bootstrap returns a fully populated JarvisApplication."""
        app = self.manager.bootstrap()
        self.assertIsInstance(app, JarvisApplication)
        self.assertIsNotNone(app.settings)
        self.assertIsNotNone(app.logger)
        self.assertIsNotNone(app.tool_registry)
        self.assertIsNotNone(app.orchestrator)
        self.assertIsNotNone(app.tts_service)
        self.assertIsNotNone(app.stt_provider)
        self.assertIsNotNone(app.wake_word_detector)
        self.assertIsNotNone(app.conversation_manager)

    def test_registered_tools_count(self) -> None:
        """Verifies that registered tools count during bootstrap includes CodingAgentTool."""
        app = self.manager.bootstrap()
        self.assertGreaterEqual(len(app.tool_registry), 11)
        self.assertIsNotNone(app.tool_registry.get_tool("coding_actions"))

    def test_coding_agent_imports_and_tool_registration(self) -> None:
        """Verifies CodingAgent components and DesktopEditorController import correctly."""
        from backend.agents.coding_agent.editor_controller import DesktopEditorController, CodingEditorController
        from backend.agents.coding_agent.coordinator import CodingAgent
        from backend.tools.coding_agent_tool import CodingAgentTool

        self.assertIs(DesktopEditorController, CodingEditorController)

        app = self.manager.bootstrap()
        self.assertIsNotNone(app.tool_registry.get_tool("coding_actions"))
        tool = app.tool_registry.get_tool("coding_actions")
        self.assertIsInstance(tool, CodingAgentTool)

    @patch("backend.core.startup.get_settings")
    def test_bootstrap_error_handling(self, mock_get_settings) -> None:
        """Verifies that initialization exceptions are caught and wrapped in ConfigurationError."""
        mock_get_settings.side_effect = RuntimeError("Settings file corrupted")
        with self.assertRaises(ConfigurationError):
            self.manager.bootstrap()


if __name__ == "__main__":
    unittest.main()
