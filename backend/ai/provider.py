"""
JARVIS AI Provider Abstract Interface & Mock Provider.

1. Why this module exists:
   Enforces Open-Closed & Dependency Inversion principles by contractually defining
   `BaseLLMProvider`. Prevents hardcoding specific AI vendors (OpenAI, Anthropic, Ollama).

2. How it fits into the architecture:
   Part of the AI abstraction layer. High-level orchestrators depend strictly on `BaseLLMProvider`.

3. Which future modules will interact with it:
   - `backend.ai.openai_provider` (Future V2+ integration)
   - `backend.ai.ollama_provider` (Future V5 integration for local LLMs)
   - `backend.services.llm_service`

4. Common mistakes to avoid:
   - Adding vendor-specific parameters directly to the base abstract class method signatures.

5. Possible future improvements:
   - Async generation methods and structured tool-calling schema translation.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("jarvis.ai.provider")


class BaseLLMProvider(ABC):
    """Abstract Base Class defining the contract for all AI LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier name of the AI provider."""

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generates a text completion response for the given user prompt.

        Args:
            prompt: User text input string.
            system_prompt: Optional instructions guiding AI persona and behavior.

        Returns:
            Generated text response string.
        """


class MockLLMProvider(BaseLLMProvider):
    """V1.0 Lightweight Mock LLM Provider for foundation testing without external API calls."""

    def __init__(self, model_name: str = "mock-v1") -> None:
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return f"MockProvider({self._model_name})"

    def generate_response(self, prompt: str, system_prompt: str | None = None) -> str:
        logger.debug("MockLLMProvider processing prompt: %s", prompt)
        return f"JARVIS V1.0 Foundation: I received your request: '{prompt}'."
