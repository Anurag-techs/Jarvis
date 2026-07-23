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

        # Step 1: Lightweight deduplication based on exact content comparison (case-insensitive)
        existing_contents = {item.content.strip().lower() for item in existing_memories}
        
        filtered_memories: list[ExtractedMemory] = []
        seen_new_contents = set()
        
        for new_mem in new_memories:
            content_clean = new_mem.content.strip()
            content_lower = content_clean.lower()
            
            # Skip if it already exists in memory, or is a duplicate within the new batch
            if content_lower in existing_contents or content_lower in seen_new_contents:
                logger.debug("Deduplicated memory during consolidation: %s", content_clean)
                continue
                
            seen_new_contents.add(content_lower)
            filtered_memories.append(new_mem)

        # For more complex/LLM-based conflict resolution, we could invoke the LLM provider here.
        # As a placeholder, we perform rule-based deduplication above which is robust for V1.
        return filtered_memories
