"""
JARVIS LLM Service.

1. Why this module exists:
   Encapsulates AI completion orchestration, prompt template formatting, and provider fallback logic.

2. How it fits into the architecture:
   Part of the Service layer. Wraps `BaseLLMProvider` abstractions to present clean text generation capabilities.

3. Which future modules will interact with it:
   - `backend.core.orchestrator.SystemOrchestrator`
   - Future agent workflow managers.

4. Common mistakes to avoid:
   - Direct vendor SDK calls without delegating to `BaseLLMProvider`.

5. Possible future improvements:
   - Dynamic prompt template loading from filesystem.
"""

import logging

from backend.ai.provider import BaseLLMProvider, MockLLMProvider

logger = logging.getLogger("jarvis.services.llm")


class LLMService:
    """Service providing managed text completions via AI providers."""

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self._provider = provider or MockLLMProvider()

    def generate_chat_response(self, user_prompt: str) -> str:
        """Generates conversational output using configured LLM provider."""
        logger.debug("LLMService delegating prompt to provider: %s", self._provider.provider_name)
        system_instructions = (
            "You are JARVIS, a production-quality AI assistant. "
            "Respond concisely, accurately, and politely."
        )
        return self._provider.generate_response(user_prompt, system_prompt=system_instructions)
