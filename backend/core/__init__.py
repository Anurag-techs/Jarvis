"""
JARVIS Core Module Package Initialization.

1. Why this module exists:
   Exposes central domain models, system exceptions, and lifecycle abstractions.
"""

from backend.core.exceptions import (
    ConfigurationError,
    JarvisError,
    ProviderError,
    ToolExecutionError,
    VoiceError,
)
from backend.core.lifecycle import BaseLifecycleManager
from backend.core.models import AssistantResponse, CommandResult, ConversationMessage, ToolCall, ToolResult, UserIntent

__all__ = [
    "BaseLifecycleManager",
    "JarvisError",
    "ConfigurationError",
    "ProviderError",
    "ToolExecutionError",
    "VoiceError",
    "UserIntent",
    "CommandResult",
    "ToolCall",
    "ToolResult",
    "AssistantResponse",
    "ConversationMessage",
]
