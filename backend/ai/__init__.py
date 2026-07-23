"""
JARVIS AI Abstraction Package.

1. Why this module exists:
   Defines abstract interfaces and provider implementations for LLM integration.
"""

from backend.ai.provider import BaseLLMProvider, MockLLMProvider
from backend.ai.providers.gemini_provider import GeminiProvider
from backend.config.settings import AIConfig

__all__ = ["BaseLLMProvider", "MockLLMProvider", "GeminiProvider", "AIConfig"]
