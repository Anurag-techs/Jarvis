"""
JARVIS Memory Summarizer.

1. Why this module exists:
   Consolidates, summarizes, or deduplicates overlapping or conflicting memory items.

2. How it fits into the architecture:
   Invoked by the MemoryPipeline coordinator to ensure we don't save duplicate facts
   and to keep memory records clean and cohesive.
"""

import logging
from backend.ai.provider import BaseLLMProvider, MockLLMProvider
from backend.memory.models import ExtractedMemory, MemoryItem

logger = logging.getLogger("jarvis.memory.summarizer")


class MemorySummarizer:
    """Consolidates or merges new and existing memories to maintain consistency."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        """Initialize the summarizer.

        Args:
            llm_provider: LLM provider instance for consolidation tasks.
        """
        self.llm_provider = llm_provider

    def consolidate(
        self, existing_memories: list[MemoryItem], new_memories: list[ExtractedMemory]
    ) -> list[ExtractedMemory]:
        """Consolidates new memories with existing memories to filter duplicates or conflicts.

        Args:
            existing_memories: Currently stored memory items.
            new_memories: Newly extracted memory facts.

        Returns:
            list[ExtractedMemory]: Consolidated/filtered list of memories to persist.
        """
        if not new_memories:
            return []

        from datetime import datetime
        now = datetime.now()

        filtered_memories: list[ExtractedMemory] = []
        seen_new_contents = set()
        
        for new_mem in new_memories:
            content_clean = new_mem.content.strip()
            content_lower = content_clean.lower()
            
            # Skip if it is a duplicate within the new batch
            if content_lower in seen_new_contents:
                logger.debug("Deduplicated memory in new batch: %s", content_clean)
                continue
                
            # Find if there is an active matching memory in storage
            active_existing = None
            for item in existing_memories:
                if (not getattr(item, "is_deleted", False)
                    and (not getattr(item, "expires_at", None) or item.expires_at >= now)
                    and item.content.strip().lower() == content_lower):
                    active_existing = item
                    break

            if active_existing:
                # If active, check if there's any update needed (e.g. importance or source changes)
                if active_existing.importance == new_mem.importance and active_existing.source == new_mem.source:
                    logger.debug("Deduplicated active memory during consolidation: %s", content_clean)
                    continue

            seen_new_contents.add(content_lower)
            filtered_memories.append(new_mem)

        return filtered_memories

