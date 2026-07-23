"""
Tests for GeminiProvider resilient error handling and retry logic.

Covers:
  - 503 UNAVAILABLE (transient → retried, then friendly fallback)
  - Timeout errors (transient → retried)
  - Network/connection errors (transient → retried)
  - Retry success on 3rd attempt
  - Retry exhaustion → friendly fallback response (JARVIS stays alive)
  - Permanent errors are NOT retried
  - Orchestrator safety net: ProviderError escaping provider is caught
  - _is_transient_error() classification
"""

import unittest
from unittest.mock import MagicMock, call, patch

from backend.ai.providers.gemini_provider import (
    GeminiProvider,
    _FRIENDLY_FALLBACK,
    _MAX_RETRIES,
    _is_transient_error,
)
from backend.config.settings import AIConfig
from backend.core.exceptions import ProviderError
from backend.core.models import AssistantResponse
from backend.core.orchestrator import SystemOrchestrator
from backend.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_config(api_key: str = "test-key-for-retry-tests") -> AIConfig:
    return AIConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=api_key,
    )


def _make_provider(mock_client: MagicMock) -> GeminiProvider:
    """Constructs a GeminiProvider with a fully mocked SDK client."""
    mock_genai = MagicMock()
    mock_types = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_google = MagicMock()
    mock_google.genai = mock_genai

    import sys
    with patch.dict(sys.modules, {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }):
        provider = GeminiProvider(config=_make_config())
        provider._client = mock_client

    return provider


def _provider_error(msg: str) -> ProviderError:
    return ProviderError(message=msg, details={})


# ---------------------------------------------------------------------------
# _is_transient_error classification
# ---------------------------------------------------------------------------

class TestTransientErrorClassification(unittest.TestCase):
    def test_503_in_message(self) -> None:
        self.assertTrue(_is_transient_error(Exception("503 UNAVAILABLE")))

    def test_429_in_message(self) -> None:
        self.assertTrue(_is_transient_error(Exception("429 Too Many Requests")))

    def test_timeout_builtin(self) -> None:
        self.assertTrue(_is_transient_error(TimeoutError("deadline exceeded")))

    def test_connection_error_builtin(self) -> None:
        self.assertTrue(_is_transient_error(ConnectionError("connection refused")))

    def test_unavailable_phrase(self) -> None:
        self.assertTrue(_is_transient_error(Exception("Service unavailable")))

    def test_resource_exhausted(self) -> None:
        self.assertTrue(_is_transient_error(Exception("resource exhausted, try again")))

    def test_permanent_error_not_transient(self) -> None:
        self.assertFalse(_is_transient_error(Exception("API key invalid")))

    def test_unknown_error_not_transient(self) -> None:
        self.assertFalse(_is_transient_error(Exception("Some unexpected error")))


# ---------------------------------------------------------------------------
# 503 UNAVAILABLE → retry then friendly fallback
# ---------------------------------------------------------------------------

class TestGeminiProvider503Handling(unittest.TestCase):
    """Verifies that a 503 UNAVAILABLE error is retried and then returns friendly fallback."""

    def setUp(self) -> None:
        self.mock_client = MagicMock()
        self.provider = _make_provider(self.mock_client)

    def test_503_causes_retry_and_friendly_fallback(self) -> None:
        """All 4 attempts (1 + 3 retries) raise 503 → friendly AssistantResponse returned."""
        error_503 = _provider_error("Gemini AI generation failed: 503 UNAVAILABLE")

        with patch.object(self.provider, "_call_api", side_effect=error_503) as mock_call, \
             patch("time.sleep") as mock_sleep:
            result = self.provider.generate_completion("Hello")

        # 4 total attempts: 1 original + 3 retries
        self.assertEqual(mock_call.call_count, _MAX_RETRIES + 1)
        # 3 sleeps: 1s, 2s, 4s
        self.assertEqual(mock_sleep.call_count, _MAX_RETRIES)
        mock_sleep.assert_has_calls([call(1.0), call(2.0), call(4.0)])

        self.assertIsInstance(result, AssistantResponse)
        self.assertEqual(result.text, _FRIENDLY_FALLBACK)
        self.assertFalse(result.success)
        self.assertTrue(result.should_speak)

    def test_503_logs_warning_per_attempt(self) -> None:
        """Each transient retry emits a WARNING log."""
        error_503 = _provider_error("Gemini AI generation failed: 503 UNAVAILABLE")

        with patch.object(self.provider, "_call_api", side_effect=error_503), \
             patch("time.sleep"), \
             self.assertLogs("jarvis.ai.providers.gemini", level="WARNING") as log_ctx:
            self.provider.generate_completion("test input")

        warning_lines = [l for l in log_ctx.output if "WARNING" in l]
        # 3 retry warnings + 1 final error log
        self.assertGreaterEqual(len(warning_lines), _MAX_RETRIES)


# ---------------------------------------------------------------------------
# Timeout error → retry then friendly fallback
# ---------------------------------------------------------------------------

class TestGeminiProviderTimeoutHandling(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = _make_provider(MagicMock())

    def test_timeout_error_is_retried(self) -> None:
        """TimeoutError exceptions are retried and produce friendly fallback."""
        timeout_error = _provider_error("Gemini AI generation failed: timed out")

        with patch.object(self.provider, "_call_api", side_effect=timeout_error), \
             patch("time.sleep") as mock_sleep:
            result = self.provider.generate_completion("What time is it?")

        self.assertEqual(result.text, _FRIENDLY_FALLBACK)
        self.assertFalse(result.success)
        self.assertEqual(mock_sleep.call_count, _MAX_RETRIES)


# ---------------------------------------------------------------------------
# Network/connection error → retry then friendly fallback
# ---------------------------------------------------------------------------

class TestGeminiProviderNetworkErrorHandling(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = _make_provider(MagicMock())

    def test_connection_error_is_retried(self) -> None:
        """Connection-related errors are retried and produce friendly fallback."""
        network_error = _provider_error("Gemini AI generation failed: connection reset by peer")

        with patch.object(self.provider, "_call_api", side_effect=network_error), \
             patch("time.sleep") as mock_sleep:
            result = self.provider.generate_completion("Tell me about Python.")

        self.assertEqual(result.text, _FRIENDLY_FALLBACK)
        self.assertFalse(result.success)
        self.assertEqual(mock_sleep.call_count, _MAX_RETRIES)


# ---------------------------------------------------------------------------
# Retry succeeds on 3rd attempt
# ---------------------------------------------------------------------------

class TestGeminiProviderRetrySuccess(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = _make_provider(MagicMock())

    def test_retry_succeeds_on_third_attempt(self) -> None:
        """After 2 transient failures, the 3rd attempt succeeds and returns real response."""
        transient = _provider_error("503 UNAVAILABLE")
        success = AssistantResponse(
            text="Python is a programming language.",
            tool_calls=[],
            should_speak=True,
            success=True,
        )

        with patch.object(
            self.provider, "_call_api",
            side_effect=[transient, transient, success]
        ) as mock_call, \
             patch("time.sleep") as mock_sleep:
            result = self.provider.generate_completion("What is Python?")

        # 3 total calls: 2 failures + 1 success
        self.assertEqual(mock_call.call_count, 3)
        # 2 sleeps: 1s then 2s
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])
        self.assertEqual(result.text, "Python is a programming language.")
        self.assertTrue(result.success)

    def test_no_sleep_on_first_success(self) -> None:
        """If first attempt succeeds, no sleep is called."""
        success = AssistantResponse(text="OK", tool_calls=[], should_speak=True, success=True)

        with patch.object(self.provider, "_call_api", return_value=success), \
             patch("time.sleep") as mock_sleep:
            result = self.provider.generate_completion("hello")

        mock_sleep.assert_not_called()
        self.assertTrue(result.success)


# ---------------------------------------------------------------------------
# Permanent error — no retry
# ---------------------------------------------------------------------------

class TestGeminiProviderPermanentErrors(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = _make_provider(MagicMock())

    def test_permanent_error_not_retried(self) -> None:
        """API key invalid is a permanent error — must not be retried."""
        permanent = _provider_error("API key invalid, check your credentials")

        with patch.object(self.provider, "_call_api", side_effect=permanent) as mock_call, \
             patch("time.sleep") as mock_sleep:
            result = self.provider.generate_completion("hello")

        # Only 1 attempt — no retry
        self.assertEqual(mock_call.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(result.text, _FRIENDLY_FALLBACK)
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# Orchestrator safety net — ProviderError escaping provider
# ---------------------------------------------------------------------------

class TestOrchestratorProviderErrorSafetyNet(unittest.TestCase):
    """Verifies the orchestrator never crashes even if ProviderError escapes the provider."""

    def setUp(self) -> None:
        self.registry = ToolRegistry()

    def test_provider_error_produces_friendly_response(self) -> None:
        """If ProviderError escapes generate_response(), orchestrator returns friendly message."""
        from backend.ai.provider import BaseLLMProvider

        failing_llm = MagicMock(spec=BaseLLMProvider)
        failing_llm.provider_name = "FailingProvider"
        failing_llm.generate_completion.side_effect = ProviderError(
            message="503 UNAVAILABLE", details={}
        )

        orchestrator = SystemOrchestrator(
            llm_provider=failing_llm,
            tool_registry=self.registry,
        )

        # Must NOT raise — must return a friendly response
        result = orchestrator.process("What is Python?")

        self.assertIsInstance(result, AssistantResponse)
        self.assertFalse(result.success)
        self.assertIn("trouble reaching", result.text.lower())

    def test_unexpected_exception_produces_friendly_response(self) -> None:
        """If any other unexpected error occurs, orchestrator returns a safe response."""
        from backend.ai.provider import BaseLLMProvider

        failing_llm = MagicMock(spec=BaseLLMProvider)
        failing_llm.provider_name = "FailingProvider"
        failing_llm.generate_completion.side_effect = RuntimeError("segfault in sdk")

        orchestrator = SystemOrchestrator(
            llm_provider=failing_llm,
            tool_registry=self.registry,
        )

        result = orchestrator.process("hello")

        self.assertIsInstance(result, AssistantResponse)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_jarvis_continues_after_provider_failure(self) -> None:
        """Simulates a real JARVIS session: one failed call followed by a successful one."""
        from backend.ai.provider import MockLLMProvider

        mock_llm = MockLLMProvider()
        orchestrator = SystemOrchestrator(
            llm_provider=mock_llm,
            tool_registry=self.registry,
        )

        # First call: simulate provider error inside conversation manager
        with patch.object(
            orchestrator._conversation_manager,
            "generate_response",
            side_effect=ProviderError(message="503 UNAVAILABLE", details={}),
        ):
            failed_result = orchestrator.process("question during outage")

        self.assertFalse(failed_result.success)
        self.assertIn("trouble", failed_result.text.lower())

        # Second call: provider recovers — should succeed normally
        normal_result = orchestrator.process("hello")
        self.assertIsNotNone(normal_result.text)
        # The loop is still alive — orchestrator did not terminate


if __name__ == "__main__":
    unittest.main()
