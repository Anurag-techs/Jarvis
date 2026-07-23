"""
Unit tests for JARVIS Configuration, Settings, and Provider Selection Logic.
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.ai.provider import MockLLMProvider
from backend.config.settings import AIConfig, Settings
from backend.core.exceptions import ProviderError
from backend.core.startup import StartupManager


class TestConfig(unittest.TestCase):
    def test_default_settings_initialization(self) -> None:
        """Verifies that Settings instantiates correctly and core fields are populated.

        Note: Settings() reads from the project .env file when present.
        This test validates structural properties rather than exact defaults
        to remain stable regardless of .env configuration.
        """
        settings = Settings()
        self.assertIsInstance(settings.assistant_name, str)
        self.assertTrue(len(settings.assistant_name) > 0)
        self.assertIsInstance(settings.wake_word, str)
        self.assertTrue(len(settings.wake_word) > 0)
        self.assertIsInstance(settings.voice_mode, bool)
        self.assertIn(settings.env, ["development", "staging", "production"])


    def test_settings_override(self) -> None:
        """Verifies that settings can be customized cleanly."""
        settings = Settings(assistant_name="CustomJARVIS", debug=False, voice_mode=True)
        self.assertEqual(settings.assistant_name, "CustomJARVIS")
        self.assertFalse(settings.debug)
        self.assertTrue(settings.voice_mode)

    def test_default_llm_provider_is_gemini(self) -> None:
        """Default llm_provider must be 'gemini', not 'mock'.

        The default should reflect production intent. Fallback to Mock only happens
        when the API key is absent or GeminiProvider raises during init.
        """
        settings = Settings()
        self.assertEqual(settings.llm_provider, "gemini")

    def test_get_ai_config_uses_gemini_api_key(self) -> None:
        """get_ai_config() must prefer gemini_api_key over llm_api_key."""
        settings = Settings(gemini_api_key="test-gemini-key", llm_api_key="other-key")
        config = settings.get_ai_config()
        self.assertEqual(config.api_key, "test-gemini-key")

    def test_get_ai_config_falls_back_to_llm_api_key(self) -> None:
        """get_ai_config() must fall back to llm_api_key when gemini_api_key is absent."""
        settings = Settings(gemini_api_key=None, llm_api_key="fallback-key")
        config = settings.get_ai_config()
        self.assertEqual(config.api_key, "fallback-key")

    def test_get_ai_config_api_key_none_when_both_absent(self) -> None:
        """get_ai_config() must produce api_key=None when no key is configured."""
        settings = Settings(gemini_api_key=None, llm_api_key=None)
        config = settings.get_ai_config()
        self.assertIsNone(config.api_key)


class TestProviderSelection(unittest.TestCase):
    """Tests for StartupManager._create_llm_provider selection logic.

    Uses Settings objects injected directly — no real network calls are made.
    GeminiProvider is patched at the import site inside startup.py.
    """

    def setUp(self) -> None:
        self.manager = StartupManager()

    def test_mock_provider_when_llm_provider_is_mock(self) -> None:
        """When LLM_PROVIDER=mock, MockLLMProvider must be selected regardless of key."""
        settings = Settings(llm_provider="mock", gemini_api_key="some-key")
        provider = self.manager._create_llm_provider(settings)
        self.assertIsInstance(provider, MockLLMProvider)
        self.assertIn("Mock", provider.provider_name)

    def test_mock_provider_when_gemini_key_is_missing(self) -> None:
        """When LLM_PROVIDER=gemini but GEMINI_API_KEY is empty, must fall back to Mock."""
        settings = Settings(llm_provider="gemini", gemini_api_key=None, llm_api_key=None)
        provider = self.manager._create_llm_provider(settings)
        self.assertIsInstance(provider, MockLLMProvider)
        self.assertIn("Mock", provider.provider_name)

    @patch("backend.core.startup.GeminiProvider")
    def test_gemini_provider_selected_when_key_present(self, mock_gemini_cls: MagicMock) -> None:
        """When LLM_PROVIDER=gemini and key is set, GeminiProvider must be instantiated."""
        fake_provider = MagicMock()
        fake_provider.provider_name = "GeminiProvider(gemini-2.5-flash)"
        mock_gemini_cls.return_value = fake_provider

        settings = Settings(llm_provider="gemini", gemini_api_key="valid-test-key")
        provider = self.manager._create_llm_provider(settings)

        mock_gemini_cls.assert_called_once()
        self.assertIs(provider, fake_provider)

    @patch("backend.core.startup.GeminiProvider")
    def test_fallback_to_mock_when_gemini_init_raises(self, mock_gemini_cls: MagicMock) -> None:
        """When GeminiProvider.__init__ raises ProviderError, must fall back to MockLLMProvider."""
        mock_gemini_cls.side_effect = ProviderError(
            message="Google Gemini API key is unconfigured. Set GEMINI_API_KEY in .env file."
        )

        settings = Settings(llm_provider="gemini", gemini_api_key="bad-or-expired-key")
        provider = self.manager._create_llm_provider(settings)

        self.assertIsInstance(provider, MockLLMProvider)
        self.assertIn("Mock", provider.provider_name)

    @patch("backend.core.startup.GeminiProvider")
    def test_provider_name_logged_correctly_for_gemini(self, mock_gemini_cls: MagicMock) -> None:
        """GeminiProvider.provider_name must NOT contain 'Mock' so startup logs are accurate."""
        fake_provider = MagicMock()
        fake_provider.provider_name = "GeminiProvider(gemini-2.5-flash)"
        mock_gemini_cls.return_value = fake_provider

        settings = Settings(llm_provider="gemini", gemini_api_key="valid-key")
        provider = self.manager._create_llm_provider(settings)

        self.assertNotIn("Mock", provider.provider_name)
        self.assertIn("Gemini", provider.provider_name)


if __name__ == "__main__":
    unittest.main()
