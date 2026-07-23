"""
JARVIS Conversation Manager.

1. Why this module exists:
   Maintains in-memory conversation context, handles history trimming, encapsulates prompt construction,
   delegates generation to AI providers, and formats tool execution feedback.

2. How it fits into the architecture:
   Part of `backend.conversation`. Injected into `SystemOrchestrator`.

3. Which future modules will interact with it:
   - `backend.core.orchestrator.SystemOrchestrator`
   - Future Memory / RAG modules (Version 3.0).

4. Common mistakes to avoid:
   - Mixing database persistence code inside in-memory ConversationManager.

5. Possible future improvements:
   - TODO: Replace message-count trimming (_trim_history) with token-based context window trimming.
"""

import logging
from typing import Any, Literal

from backend.ai.provider import BaseLLMProvider
from backend.conversation.system_prompt import BaseSystemPromptProvider, DefaultSystemPromptProvider
from backend.core.models import AssistantResponse, ConversationMessage, ToolResult

logger = logging.getLogger("jarvis.conversation.manager")


class ConversationManager:
    """Manages multi-turn conversation state, history limits, and LLM response generation."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        history_limit: int = 20,
        session_id: str = "default_session",
        system_prompt_provider: BaseSystemPromptProvider | None = None,
    ) -> None:
        """Initialize ConversationManager via dependency injection.

        Args:
            llm_provider: Abstract BaseLLMProvider instance.
            history_limit: Maximum number of conversation message turns to retain.
            session_id: Optional session identifier for multi-session support.
            system_prompt_provider: Optional provider supplying system instructions.
        """
        self._llm = llm_provider
        self._history_limit = history_limit
        self.session_id = session_id
        self._system_prompt_provider = system_prompt_provider or DefaultSystemPromptProvider()
        self._history: list[ConversationMessage] = []

        logger.info(
            "ConversationManager initialized (Session: %s, History Limit: %d)",
            self.session_id,
            self._history_limit,
        )

    def add_user_message(self, content: str) -> ConversationMessage:
        """Appends a user message turn to conversation history."""
        return self._store_message(role="user", content=content)

    def add_assistant_message(self, content: str) -> ConversationMessage:
        """Appends an assistant message turn to conversation history."""
        return self._store_message(role="assistant", content=content)

    def get_history(self) -> list[ConversationMessage]:
        """Returns a copy of the current conversation message history."""
        return list(self._history)

    def clear(self) -> None:
        """Clears all conversation history for the current session."""
        self._history.clear()
        logger.info("Cleared conversation history for session: %s", self.session_id)

    def generate_response(
        self, user_prompt: str, available_tools: list[dict[str, Any]] | None = None
    ) -> AssistantResponse:
        """Processes user prompt, manages history context, calls LLM, and returns AssistantResponse.

        Args:
            user_prompt: Raw text query from user.
            available_tools: Optional tool metadata schemas.

        Returns:
            AssistantResponse model containing text message and optional tool_calls.
        """
        logger.debug("ConversationManager processing prompt for session '%s'", self.session_id)

        # 1. Store incoming user message
        self.add_user_message(user_prompt)

        # 2. Get system prompt instructions
        system_prompt = self._system_prompt_provider.get_system_prompt()

        # 3. Build formatted history list for provider
        history_dicts = self._build_prompt()

        # 4. Generate structured completion via LLM provider
        response: AssistantResponse = self._llm.generate_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            history=history_dicts,
            available_tools=available_tools,
        )

        # 5. Store assistant completion response text
        self.add_assistant_message(response.text)

        # 6. Trim history to remain within configured bounds
        self._trim_history()

        response.metadata["session_id"] = self.session_id
        response.metadata["history_count"] = len(self._history)
        return response

    def generate_final_response(self, tool_results: list[ToolResult]) -> AssistantResponse:
        """Sends executed tool results back to LLM to generate a final natural-language response.

        Args:
            tool_results: List of executed ToolResult objects.

        Returns:
            AssistantResponse containing final natural language response text.
        """
        summary_lines = [f"- {res.message}" for res in tool_results if res.message]
        summary_prompt = f"Tool Execution Results:\n" + "\n".join(summary_lines) + "\n\nSynthesize a polite, friendly final response for the user."

        system_prompt = self._system_prompt_provider.get_system_prompt()
        history_dicts = self._build_prompt()

        final_response: AssistantResponse = self._llm.generate_completion(
            user_prompt=summary_prompt,
            system_prompt=system_prompt,
            history=history_dicts,
        )

        self.add_assistant_message(final_response.text)
        self._trim_history()
        return final_response

    def _store_message(self, role: Literal["user", "assistant", "system"], content: str) -> ConversationMessage:
        """Internal helper to create and append a ConversationMessage."""
        msg = ConversationMessage(role=role, content=content)
        self._history.append(msg)
        return msg

    def _trim_history(self) -> None:
        """Trims message history list when count exceeds configured history limit.

        TODO: In future versions, upgrade this message-count trimming to token-based context window trimming
        using tiktoken / model tokenizer counters.
        """
        if len(self._history) > self._history_limit:
            overflow = len(self._history) - self._history_limit
            logger.debug("Trimming %d oldest messages from session '%s'", overflow, self.session_id)
            self._history = self._history[-self._history_limit :]

    def _build_prompt(self) -> list[dict[str, str]]:
        """Internal helper formatting history messages into dictionary list for LLM providers."""
        return [{"role": msg.role, "content": msg.content} for msg in self._history]
