"""
Unit tests for JARVIS Configuration and Settings.
"""

import unittest
from backend.config.settings import Settings


class TestConfig(unittest.TestCase):
    def test_default_settings_initialization(self) -> None:
        """Verifies that default settings instantiate with expected defaults."""
        settings = Settings()
        self.assertEqual(settings.assistant_name, "JARVIS")
        self.assertEqual(settings.wake_word, "jarvis")
        self.assertIn(settings.env, ["development", "staging", "production"])

    def test_settings_override(self) -> None:
        """Verifies that settings can be customized cleanly."""
        settings = Settings(assistant_name="CustomJARVIS", debug=False)
        self.assertEqual(settings.assistant_name, "CustomJARVIS")
        self.assertFalse(settings.debug)


if __name__ == "__main__":
    unittest.main()
