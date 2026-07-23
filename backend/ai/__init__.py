"""
JARVIS AI Abstraction Package.

1. Why this module exists:
   Defines abstract interfaces and provider implementations for LLM integration.

2. How it fits into the architecture:
   Isolates AI vendor logic behind abstract interfaces so providers can be swapped cleanly.

3. Which future modules will interact with it:
   `backend.services.llm_service` and `backend.core.orchestrator`.

4. Common mistakes to avoid:
   Importing provider-specific SDKs (e.g. `import openai`) inside core orchestrator code.

5. Possible future improvements:
   Streaming token generators and fallback provider chains.
"""

from backend.ai.provider import BaseLLMProvider, MockLLMProvider

__all__ = ["BaseLLMProvider", "MockLLMProvider"]
