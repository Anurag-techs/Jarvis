"""
JARVIS Memory Recall Service.

1. Why this module exists:
   Encapsulates prompt context-injection and memory search orchestration.
   Keeps SystemOrchestrator and ConversationManager decoupled from the details of prompt construction.
"""

import logging
from backend.memory.base import BaseMemoryStore

logger = logging.getLogger("jarvis.memory.recall")


class MemoryRecallService:
    """Service responsible for searching past memories and injecting them as context into user prompts."""

    def __init__(self, memory_store: BaseMemoryStore) -> None:
        """Initialize the recall service with a concrete memory store.

        Args:
            memory_store: Injected BaseMemoryStore instance.
        """
        self._memory_store = memory_store

    def build_context(self, user_query: str, top_k: int = 5) -> str:
        """Searches memory for relevant context and injects it into the prompt.

        Args:
            user_query: Raw user message.
            top_k: Number of relevant memories to retrieve.

        Returns:
            The context-injected prompt or the raw query if no memories are found or an error occurs.
        """
        try:
            logger.info("Searching memory...")
            memories = self._memory_store.search_relevant_facts(user_query, top_k=top_k)
            
            if memories:
                logger.info("Found %d relevant memories.", len(memories))
                formatted = (
                    "Relevant memories:\n\n"
                    + "\n".join(f"- {mem}" for mem in memories)
                    + "\n\nCurrent user message:\n\n"
                    + user_query
                )
                logger.info("Injected memory context into prompt.")
                return formatted
            else:
                logger.info("No relevant memories found.")
        except Exception as exc:
            logger.error("Failed to retrieve relevant memories: %s", exc, exc_info=True)

        return user_query
