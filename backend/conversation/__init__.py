"""
JARVIS Conversation Package Initialization.

1. Why this module exists:
   Exposes ConversationManager and SystemPromptProvider abstractions.
"""

from backend.conversation.manager import ConversationManager
from backend.conversation.system_prompt import BaseSystemPromptProvider, DefaultSystemPromptProvider

__all__ = [
    "ConversationManager",
    "BaseSystemPromptProvider",
    "DefaultSystemPromptProvider",
]
