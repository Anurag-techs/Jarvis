"""
JARVIS System Prompt Provider Abstraction.

1. Why this module exists:
   Decouples system prompt management from the core orchestrator and conversation manager.
   Allows persona instructions, rules, and system capabilities to be updated or customized independently.

2. How it fits into the architecture:
   Part of the Conversation layer. Subclassed by prompt providers or template loaders.

3. Which future modules will interact with it:
   - `backend.conversation.manager.ConversationManager`
   - Future persona configuration modules.

4. Common mistakes to avoid:
   - Hardcoding system prompt strings inside orchestrators or LLM providers.

5. Possible future improvements:
   - Dynamic prompt template rendering with jinja2 / localized context injection.
"""

from abc import ABC, abstractmethod


class BaseSystemPromptProvider(ABC):
    """Abstract Base Class for generating system instruction prompts."""

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Returns the system instruction prompt string for the AI assistant."""


class DefaultSystemPromptProvider(BaseSystemPromptProvider):
    """Default system prompt provider returning core JARVIS persona instructions."""

    def __init__(self, assistant_name: str = "JARVIS") -> None:
        self._assistant_name = assistant_name

    def get_system_prompt(self) -> str:
        return (
            f"You are {self._assistant_name}, a production-quality AI assistant. "
            "Respond accurately, concisely, and politely. Always maintain a professional, helpful persona."
        )
