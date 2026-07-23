"""
JARVIS AI Provider Abstract Interface & Mock Provider.

1. Why this module exists:
   Enforces Open-Closed & Dependency Inversion principles by contractually defining
   `BaseLLMProvider`. Parses JSON and returns `AssistantResponse` directly from the provider layer.

2. How it fits into the architecture:
   Part of the AI abstraction layer. High-level orchestrators depend strictly on `BaseLLMProvider`.

3. Which future modules will interact with it:
   - `backend.ai.providers.gemini_provider.GeminiProvider`
   - `backend.conversation.manager.ConversationManager`

4. Common mistakes to avoid:
   - Returning raw unparsed JSON strings to the orchestrator layer.

5. Possible future improvements:
   - Native GenAI SDK Function Declaration bindings.
"""

from abc import ABC, abstractmethod
import json
import logging
from typing import Any

from backend.core.models import AssistantResponse, ToolCall

logger = logging.getLogger("jarvis.ai.provider")


class BaseLLMProvider(ABC):
    """Abstract Base Class defining the contract for all AI LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier name of the AI provider."""

    @abstractmethod
    def generate_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> AssistantResponse:
        """Generates structured AssistantResponse directly from the AI provider.

        Args:
            user_prompt: User query or prompt string.
            system_prompt: Optional instructions guiding AI persona and system behavior.
            history: Optional list of past conversation turns.
            available_tools: Optional tool metadata schema definitions.

        Returns:
            AssistantResponse DTO populated with text message and optional tool_calls.
        """

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Backward-compatible helper returning completion text string."""
        completion = self.generate_completion(user_prompt, system_prompt, history)
        return completion.text

    @abstractmethod
    def generate_raw_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        response_mime_type: str = "text/plain",
    ) -> str:
        """Generates raw text response directly from the AI provider.

        Useful for system-internal requests like memory extraction, consolidation, etc.,
        where standard AssistantResponse packaging or tool calls are not desired.
        """


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM Provider returning structured AssistantResponse for testing and offline fallback."""

    RESPONSE_MAPPING: dict[str, str] = {
        "hello": "Hello! I am JARVIS.",
        "hi": "Hello! I am JARVIS.",
        "greetings": "Hello! I am JARVIS.",
        "what is your name?": "My name is JARVIS.",
        "what is your name": "My name is JARVIS.",
        "who are you?": "My name is JARVIS.",
        "who are you": "My name is JARVIS.",
    }
    DEFAULT_RESPONSE: str = "I'm still under development."

    def __init__(self, model_name: str = "mock-v1") -> None:
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return f"MockProvider({self._model_name})"

    def generate_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> AssistantResponse:
        logger.debug("MockLLMProvider generating completion for user prompt: '%s'", user_prompt)
        lowered = user_prompt.strip().lower()

        # Synthesis pass: tool result prompts must never be returned verbatim to the user.
        # ConversationManager.generate_final_response() sends "Tool Execution Results:\n- <msg>\n\nSynthesize..."
        # The MockLLMProvider must consume this and produce a clean natural-language reply.
        if user_prompt.strip().startswith("Tool Execution Results"):
            # Extract the first result bullet as a concise summary
            lines = user_prompt.splitlines()
            result_lines = [l.lstrip("- ").strip() for l in lines if l.strip().startswith("-")]
            summary = result_lines[0] if result_lines else "I've completed the requested action."
            return AssistantResponse(
                text=summary,
                tool_calls=[],
                should_speak=True,
                success=True,
            )

        # Handle tool call decision matching for foundation testing
        if "weather" in lowered:
            location = "London"
            if "in " in lowered:
                idx = lowered.find("in ")
                location = user_prompt[idx + 3 :].strip("? .!")
            return AssistantResponse(
                text=f"Checking weather for {location}.",
                tool_calls=[ToolCall(tool="get_weather", arguments={"location": location, "query": user_prompt})],
            )

        if "open" in lowered and "website" in lowered:
            return AssistantResponse(
                text="Opening requested website.",
                tool_calls=[ToolCall(tool="open_website", arguments={"query": user_prompt})],
            )

        if "open" in lowered or "launch" in lowered:
            app = user_prompt.replace("open", "").replace("launch", "").strip()
            return AssistantResponse(
                text=f"Opening application '{app}'.",
                tool_calls=[ToolCall(tool="open_application", arguments={"app_name": app, "query": user_prompt})],
            )

        # Standard conversation response mapping
        response_text = self.RESPONSE_MAPPING.get(lowered, self.DEFAULT_RESPONSE)
        return AssistantResponse(
            text=response_text,
            tool_calls=[],
            should_speak=True,
        )

    def generate_raw_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        response_mime_type: str = "text/plain",
    ) -> str:
        logger.debug("MockLLMProvider generating raw completion for user prompt: '%s'", user_prompt)
        lowered = user_prompt.lower()
        if "user input:" in lowered and "assistant response:" in lowered:
            # Deterministic memory extraction for test simulations
            extracted = []
            if "my name is" in lowered:
                parts = user_prompt.split("my name is")
                if len(parts) > 1:
                    name = parts[1].split("\n")[0].strip("? .!").title()
                    extracted.append({
                        "content": f"User's name is {name}.",
                        "importance": 5.0,
                        "source": "user",
                        "metadata": {"category": "user_profile"}
                    })
            elif "i study" in lowered:
                parts = user_prompt.split("i study")
                if len(parts) > 1:
                    subject = parts[1].split("\n")[0].strip("? .!")
                    extracted.append({
                        "content": f"User studies {subject}.",
                        "importance": 4.0,
                        "source": "user",
                        "metadata": {"category": "user_profile"}
                    })
            return json.dumps(extracted)

        return "Mock raw response."

