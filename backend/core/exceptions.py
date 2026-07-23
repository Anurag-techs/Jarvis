"""
JARVIS Core Exception Hierarchy.

1. Why this module exists:
   Establishes a strong domain exception hierarchy so errors across AI, voice, system tools,
   and configuration can be caught, handled, and logged with precision without unhandled crashes.

2. How it fits into the architecture:
   Part of the Core domain. All custom exceptions inherit from `JarvisError`. High-level orchestrators
   and services catch these exceptions to output user-friendly responses.

3. Which future modules will interact with it:
   - backend.tools.* (Raises `ToolExecutionError`)
   - backend.ai.* (Raises `ProviderError`)
   - backend.voice.* (Raises `VoiceError`)
   - backend.config.* (Raises `ConfigurationError`)
   - backend.api.* (Converts domain exceptions into clean HTTP/WebSocket error codes)

4. Common mistakes to avoid:
   - Bare `except Exception:` blocks that swallow details.
   - Raising generic built-in Python exceptions (`ValueError`, `RuntimeError`) for domain errors.

5. Possible future improvements:
   - Error code enumeration mapping for standardized internationalization & localization.
"""


class JarvisError(Exception):
    """Base class for all exceptions originating within JARVIS domain."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(JarvisError):
    """Raised when environment configuration or settings are missing or invalid."""


class ProviderError(JarvisError):
    """Raised when an external AI provider or API fails or returns an error."""


class ToolExecutionError(JarvisError):
    """Raised when execution of a tool action fails."""


class VoiceError(JarvisError):
    """Raised when STT, TTS, or wake-word engines encounter an operational error."""
