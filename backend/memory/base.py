"""
JARVIS Memory Abstract Interface.

1. Why this module exists:
   Defines the contract for short-term and long-term memory operations.
   Per Version 1.0 guidelines, ONLY the interface is defined here. No concrete memory engine is implemented.

2. How it fits into the architecture:
   Sits in the Memory abstraction layer. The core orchestrator receives a `BaseMemoryStore`
   interface via dependency injection.

3. Which future modules will interact with it:
   - `backend.memory.vector_store` (Version 3.0 vector memory)
   - `backend.memory.relational_store` (Version 3.0 user preference memory)

4. Common mistakes to avoid:
   - Adding concrete database connection strings or storage code in V1.0.

5. Possible future improvements:
   - Hybrid graph and vector retrieval search signatures.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseMemoryStore(ABC):
    """Abstract interface for storing and retrieving conversational context and long-term facts."""

    @abstractmethod
    def store_interaction(self, user_query: str, assistant_response: str, metadata: dict[str, Any] | None = None) -> None:
        """Saves a conversation turn to context memory."""

    @abstractmethod
    def get_recent_context(self, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieves recent conversation history."""

    @abstractmethod
    def search_relevant_facts(self, query: str, top_k: int = 3) -> list[str]:
        """Performs semantic search across long-term memory facts."""
