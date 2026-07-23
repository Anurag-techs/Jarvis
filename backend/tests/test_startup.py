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
        """Verifies that 7 tools are registered during bootstrap."""
        app = self.manager.bootstrap()
        self.assertEqual(len(app.tool_registry), 7)

    @patch("backend.core.startup.get_settings")
    def test_bootstrap_error_handling(self, mock_get_settings) -> None:
        """Verifies that initialization exceptions are caught and wrapped in ConfigurationError."""
        mock_get_settings.side_effect = RuntimeError("Settings file corrupted")
        with self.assertRaises(ConfigurationError):
            self.manager.bootstrap()


if __name__ == "__main__":
    unittest.main()
