"""
JARVIS Google Gemini AI Provider.

1. Why this module exists:
   Integrates Google Gemini models (`gemini-2.5-flash`) via the official `google-genai` SDK.
   Parses structured JSON responses directly in the provider layer and returns `AssistantResponse`.

2. How it fits into the architecture:
   Concrete subclass of `BaseLLMProvider`. Injected into `SystemOrchestrator` via `StartupManager`.

3. Which future modules will interact with it:
   - `backend.core.startup.StartupManager`
   - `backend.conversation.manager.ConversationManager`

4. Common mistakes to avoid:
   - Returning raw JSON strings to orchestrators instead of parsing into `AssistantResponse`.
   - Exposing SDK-specific errors directly to callers.
   - Allowing unhandled provider exceptions to crash the conversation loop.

5. Possible future improvements:
   - Native Gemini FunctionDeclaration tool calling bindings.
   - Per-model retry budget configuration.
"""

import json
import logging
import time
from typing import Any

from backend.ai.provider import BaseLLMProvider
from backend.config.settings import AIConfig
from backend.core.exceptions import ProviderError
from backend.core.models import AssistantResponse, ToolCall

logger = logging.getLogger("jarvis.ai.providers.gemini")

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 1.0   # seconds; doubles on each attempt
_FRIENDLY_FALLBACK: str = (
    "I'm having trouble reaching the AI service right now. "
    "Please try again in a moment."
)

# HTTP status codes / substrings that indicate a transient (retriable) error
_TRANSIENT_STATUS_CODES: frozenset[str] = frozenset({"429", "500", "502", "503", "504"})
_TRANSIENT_SUBSTRINGS: tuple[str, ...] = (
    "unavailable",
    "overloaded",
    "quota",
    "rate limit",
    "timeout",
    "timed out",
    "connection",
    "reset by peer",
    "temporarily",
    "try again",
    "resource exhausted",
)


def _is_transient_error(exc: Exception) -> bool:
    """Returns True if the exception represents a transient API error worth retrying.

    Checks the exception type hierarchy (including httpx, requests, and google-genai APIErrors),
    recursively inspects causes/contexts, and searches the stringified error message for known
    transient status codes and phrases.
    """
    curr = exc
    while curr is not None:
        # Check standard transient exception classes
        if isinstance(curr, (TimeoutError, ConnectionError, ConnectionResetError)):
            return True

        # Check google.genai.errors if imported/available
        try:
            import google.genai.errors as genai_errors
            if isinstance(curr, genai_errors.APIError):
                if curr.code in (429, 500, 502, 503, 504) or str(curr.code) in _TRANSIENT_STATUS_CODES:
                    return True
        except ImportError:
            pass

        # Check httpx exception classes
        try:
            import httpx
            if isinstance(curr, (httpx.HTTPError, httpx.TimeoutException, httpx.NetworkError)):
                return True
        except ImportError:
            pass

        # Check requests exception classes
        try:
            import requests
            if isinstance(curr, requests.RequestException):
                return True
        except ImportError:
            pass

        # Check string representation of the exception message
        lowered = str(curr).lower()
        for code in _TRANSIENT_STATUS_CODES:
            if code in lowered:
                return True
        if any(phrase in lowered for phrase in _TRANSIENT_SUBSTRINGS):
            return True

        # Traverse context / cause for wrapped exceptions
        if getattr(curr, "__cause__", None) is not None:
            curr = curr.__cause__
        elif getattr(curr, "__context__", None) is not None:
            curr = curr.__context__
        else:
            break

    return False


class GeminiProvider(BaseLLMProvider):
    """Google Gemini AI Provider with exponential-backoff retry for transient errors."""

    def __init__(self, config: AIConfig) -> None:
        """Initializes the Gemini client instance once using provided AIConfig.

        Args:
            config: Encapsulated AI provider configuration object.

        Raises:
            ProviderError: If API key is missing or SDK client initialization fails.
        """
        self._config = config
        self._model_name = config.model_name

        if not config.api_key:
            raise ProviderError(
                message="Google Gemini API key is unconfigured. Set GEMINI_API_KEY in .env file.",
                details={"provider": "gemini", "model": self._model_name},
            )

        try:
            import google.genai as genai  # Deferred import for clean SDK loading

            # Single client initialization reused across all requests
            self._client = genai.Client(api_key=config.api_key)
        except ImportError as exc:
            raise ProviderError(
                message=(
                    "google-genai SDK is not installed. "
                    "Run: pip install google-genai>=1.0.0"
                ),
                details={"provider": "gemini", "model": self._model_name},
            ) from exc
        except Exception as exc:
            raise ProviderError(
                message=f"Failed to initialize Google GenAI SDK client: {exc}",
                details={"provider": "gemini", "model": self._model_name},
            ) from exc

    @property
    def provider_name(self) -> str:
        return f"GeminiProvider({self._model_name})"

    def generate_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> AssistantResponse:
        """Generates structured completion with automatic retry for transient errors.

        Implements exponential backoff: delay = base * 2^attempt (1s, 2s, 4s).
        Transient errors (503, timeout, connection errors) are retried up to
        _MAX_RETRIES times. Permanent errors fail immediately.
        After all retries are exhausted, returns a friendly AssistantResponse
        instead of raising — JARVIS never crashes because of an LLM failure.

        Args:
            user_prompt: Target query string from user.
            system_prompt: Optional instructions guiding AI persona and system behavior.
            history: Optional list of past conversation turns.
            available_tools: Optional tool metadata schemas.

        Returns:
            AssistantResponse DTO. On permanent failure, returns a friendly error message
            with success=False so callers can remain alive.
        """
        if not user_prompt or not user_prompt.strip():
            return AssistantResponse(
                text="I did not receive a prompt to generate.",
                should_speak=False,
                success=False,
            )

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._call_api(user_prompt, system_prompt, available_tools)

            except ProviderError as exc:
                # Only retry transient ProviderErrors (those wrapping 503/timeout/etc.)
                if not _is_transient_error(exc):
                    # Permanent error (e.g. invalid key, malformed request) — fail fast
                    logger.error(
                        "Gemini permanent error on attempt %d/%d: %s",
                        attempt + 1, _MAX_RETRIES + 1, exc.message,
                        exc_info=True,
                    )
                    return self._friendly_fallback(user_prompt, exc)

                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Gemini transient error on attempt %d/%d (%s). "
                        "Retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES + 1, exc.message, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Gemini transient error after %d/%d attempts (%s). "
                        "All retries exhausted.",
                        attempt + 1, _MAX_RETRIES + 1, exc.message,
                        exc_info=True,
                    )

        return self._friendly_fallback(user_prompt, last_exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None,
        available_tools: list[dict[str, Any]] | None,
    ) -> AssistantResponse:
        """Single attempt at calling the Gemini API. Raises ProviderError on any failure."""
        try:
            import google.genai.types as types

            full_system_instruction = self._build_system_instruction(system_prompt, available_tools)

            request_config = types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
                response_mime_type="application/json",
            )

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=user_prompt,
                config=request_config,
            )

            if not response or not response.text:
                raise ProviderError(
                    message="Gemini API returned an empty completion response.",
                    details={"model": self._model_name},
                )

            return self._parse_json_response(response.text.strip())

        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                message=f"Gemini AI generation failed: {exc}",
                details={"provider": "gemini", "model": self._model_name, "prompt": user_prompt},
            ) from exc

    def _friendly_fallback(self, user_prompt: str, exc: Exception | None) -> AssistantResponse:
        """Returns a user-friendly AssistantResponse when all retries are exhausted."""
        logger.error(
            "Returning friendly fallback response for prompt=%r after provider failure.",
            user_prompt[:80],
            exc_info=exc is not None,
        )
        return AssistantResponse(
            text=_FRIENDLY_FALLBACK,
            tool_calls=[],
            should_speak=True,
            success=False,
            error=str(exc) if exc else "Unknown provider error",
        )

    def _build_system_instruction(
        self, base_system_prompt: str | None, available_tools: list[dict[str, Any]] | None
    ) -> str:
        """Formats the system prompt with tool metadata schemas and JSON schema instructions."""
        prompt_parts = [base_system_prompt or "You are JARVIS, a production-quality AI assistant."]

        if available_tools:
            tools_json = json.dumps(available_tools, indent=2)
            prompt_parts.append(
                f"\nAVAILABLE TOOLS:\n{tools_json}\n\n"
                "INSTRUCTIONS:\n"
                "If the user query requires executing a tool, include the tool name and arguments in 'tool_calls'.\n"
                "Otherwise, set 'tool_calls' to an empty list [].\n\n"
                "REQUIRED JSON OUTPUT SCHEMA:\n"
                "{\n"
                '  "message": "Human readable text summary",\n'
                '  "tool_calls": [\n'
                '    {"tool": "tool_name", "arguments": {"arg": "value"}}\n'
                "  ]\n"
                "}"
            )
        else:
            prompt_parts.append(
                "\nREQUIRED JSON OUTPUT SCHEMA:\n"
                '{\n  "message": "Response text",\n  "tool_calls": []\n}'
            )

        return "\n\n".join(prompt_parts)

    def _parse_json_response(self, raw_text: str) -> AssistantResponse:
        """Parses raw JSON output into AssistantResponse model."""
        try:
            payload = json.loads(raw_text)
            message = payload.get("message", raw_text)
            tool_calls_data = payload.get("tool_calls", [])

            parsed_tool_calls: list[ToolCall] = []
            for call_item in tool_calls_data:
                tool_name = call_item.get("tool") or call_item.get("name")
                args = call_item.get("arguments") or call_item.get("args") or {}
                if tool_name:
                    parsed_tool_calls.append(ToolCall(tool=tool_name, arguments=args))

            return AssistantResponse(
                text=message,
                tool_calls=parsed_tool_calls,
                should_speak=True,
                success=True,
            )
        except Exception as exc:
            logger.warning("Failed to parse Gemini JSON output (%s); returning text directly", exc)
            return AssistantResponse(text=raw_text, tool_calls=[], should_speak=True)

    def generate_raw_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        response_mime_type: str = "text/plain",
    ) -> str:
        """Generates raw text or JSON response directly from Gemini with retry resilience."""
        if not user_prompt or not user_prompt.strip():
            return ""

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._call_raw_api(user_prompt, system_prompt, response_mime_type)

            except ProviderError as exc:
                if not _is_transient_error(exc):
                    logger.error(
                        "Gemini permanent error on raw attempt %d/%d: %s",
                        attempt + 1, _MAX_RETRIES + 1, exc.message,
                        exc_info=True,
                    )
                    raise
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Gemini transient error on raw attempt %d/%d (%s). "
                        "Retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES + 1, exc.message, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Gemini transient error after raw %d/%d attempts (%s). "
                        "All retries exhausted.",
                        attempt + 1, _MAX_RETRIES + 1, exc.message,
                        exc_info=True,
                    )

        raise ProviderError(
            message=f"Gemini AI raw generation failed after retries: {last_exc}",
            details={"provider": "gemini", "model": self._model_name},
        ) from last_exc

    def _call_raw_api(
        self,
        user_prompt: str,
        system_prompt: str | None,
        response_mime_type: str,
    ) -> str:
        """Single attempt at calling the Gemini API directly. Raises ProviderError on failure."""
        try:
            import google.genai.types as types

            request_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
                response_mime_type=response_mime_type,
            )

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=user_prompt,
                config=request_config,
            )

            if not response or not response.text:
                raise ProviderError(
                    message="Gemini API returned an empty raw response.",
                    details={"model": self._model_name},
                )

            return response.text.strip()

        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                message=f"Gemini AI raw generation failed: {exc}",
                details={"provider": "gemini", "model": self._model_name, "prompt": user_prompt},
            ) from exc
