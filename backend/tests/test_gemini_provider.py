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
                provider.generate_response("Hello")

            self.assertIn("Gemini AI generation failed", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
