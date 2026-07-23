"""
Unit & Integration tests for Google Gemini Provider (Sprint 3).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from backend.config.settings import AIConfig
from backend.core.exceptions import ProviderError
from backend.ai.providers.gemini_provider import GeminiProvider


class TestGeminiProvider(unittest.TestCase):
    def test_missing_api_key_raises_provider_error(self) -> None:
        """Verifies GeminiProvider raises ProviderError if GEMINI_API_KEY is unconfigured."""
        config = AIConfig(provider="gemini", model_name="gemini-2.5-flash", api_key=None)
        with self.assertRaises(ProviderError) as ctx:
            GeminiProvider(config=config)

        self.assertIn("API key is unconfigured", ctx.exception.message)

    def test_gemini_client_initialization_and_generation(self) -> None:
        """Verifies GeminiProvider initializes client once and generates completion using mocked SDK."""
        config = AIConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key="test-secret-key-12345",
            temperature=0.7,
            max_tokens=1024,
        )

        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from Google Gemini 2.5 Flash!"

        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        mock_google = MagicMock()
        mock_google.genai = mock_genai
        mock_google.genai.types = mock_types

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            provider = GeminiProvider(config=config)
            self.assertEqual(provider.provider_name, "GeminiProvider(gemini-2.5-flash)")
            mock_genai.Client.assert_called_once_with(api_key="test-secret-key-12345")

            result = provider.generate_response(
                user_prompt="Who created you?",
                system_prompt="You are JARVIS.",
                history=[{"user": "Hi", "assistant": "Hello"}],
            )

            self.assertEqual(result, "Hello from Google Gemini 2.5 Flash!")
            mock_client.models.generate_content.assert_called_once()

    def test_gemini_provider_wraps_sdk_exceptions(self) -> None:
        """Verifies GeminiProvider wraps SDK-specific exceptions into ProviderError."""
        config = AIConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key="test-secret-key-12345",
        )

        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.side_effect = RuntimeError("429 Quota Exceeded")

        mock_google = MagicMock()
        mock_google.genai = mock_genai
        mock_google.genai.types = mock_types

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            provider = GeminiProvider(config=config)
            with self.assertRaises(ProviderError) as ctx:
                provider._call_api("Hello", None, None)

            self.assertIn("Gemini AI generation failed", ctx.exception.message)

    def test_gemini_provider_raw_completion(self) -> None:
        """Verifies GeminiProvider generates raw content successfully using mocked SDK."""
        config = AIConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key="test-secret-key-12345",
        )

        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Raw content from Google Gemini."

        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        mock_google = MagicMock()
        mock_google.genai = mock_genai
        mock_google.genai.types = mock_types

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            provider = GeminiProvider(config=config)
            result = provider.generate_raw_completion(
                user_prompt="Raw test prompt",
                system_prompt="Raw test system instruction",
                response_mime_type="text/plain"
            )
            self.assertEqual(result, "Raw content from Google Gemini.")
            mock_client.models.generate_content.assert_called_once()

    def test_gemini_sdk_not_installed_raises_provider_error_with_install_hint(self) -> None:
        """Verifies that an ImportError (SDK not installed) is wrapped as ProviderError with install hint.

        This is the regression test for the bug where `No module named 'google'` was caught
        as a generic exception and silently fell back to MockProvider with no actionable message.
        """
        config = AIConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key="test-key",
        )

        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            with self.assertRaises(ProviderError) as ctx:
                GeminiProvider(config=config)

        self.assertIn("pip install", ctx.exception.message)
        self.assertIn("google-genai", ctx.exception.message)


class TestProviderSelectionWithRealSDK(unittest.TestCase):
    """Regression tests verifying that _create_llm_provider selects GeminiProvider
    when the google-genai SDK IS installed and a valid key is configured.

    These tests use the real installed SDK — patching only the network call (Client).
    """

    def test_gemini_selected_when_sdk_installed_and_key_present(self) -> None:
        """When google-genai is installed and GEMINI_API_KEY is set, GeminiProvider must be selected.

        This is the primary regression test for the 'MockProvider used despite valid key' bug,
        which was caused by google-genai not being installed in the venv.
        """
        import importlib
        from backend.config.settings import AIConfig, Settings
        from backend.ai.providers.gemini_provider import GeminiProvider
        from backend.core.startup import StartupManager

        # Verify the SDK is available before running this test
        genai_spec = importlib.util.find_spec("google.genai")
        if genai_spec is None:
            self.skipTest("google-genai SDK is not installed — install with: pip install google-genai>=1.0.0")

        # Patch only the network-bound Client constructor; keep real import path
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()

            settings = Settings(llm_provider="gemini", gemini_api_key="test-regression-key")
            manager = StartupManager()
            provider = manager._create_llm_provider(settings)

        # Must be a GeminiProvider, NOT MockLLMProvider
        self.assertIsInstance(provider, GeminiProvider)
        self.assertIn("Gemini", provider.provider_name)
        self.assertNotIn("Mock", provider.provider_name)

    def test_mock_selected_when_sdk_installed_but_key_absent(self) -> None:
        """When SDK is installed but GEMINI_API_KEY is absent, MockLLMProvider must be selected."""
        from backend.ai.provider import MockLLMProvider
        from backend.config.settings import Settings
        from backend.core.startup import StartupManager

        settings = Settings(llm_provider="gemini", gemini_api_key=None, llm_api_key=None)
        manager = StartupManager()
        provider = manager._create_llm_provider(settings)

        self.assertIsInstance(provider, MockLLMProvider)


if __name__ == "__main__":
    unittest.main()
