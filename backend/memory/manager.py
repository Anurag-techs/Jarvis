"""
JARVIS Memory Storage Manager.

1. Why this module exists:
   Encapsulates physical storage operations for memory items (remember, recall, forget, search, clear).
   Ensures single-responsibility principle by only managing storage state.

2. How it fits into the architecture:
   Acts as the storage layer for the MemoryPipeline coordinator.
   Per Version 1.0 architecture guidelines, this uses in-memory persistence.
"""

from datetime import datetime
import logging
from typing import Any

from backend.memory.models import MemoryItem, SearchType

logger = logging.getLogger("jarvis.memory.manager")


class MemoryManager:
    """Manages storage, retrieval, search, deletion, and cleanup of MemoryItems."""

    def __init__(self) -> None:
        """Initialize in-memory database store."""
        self._store: dict[str, MemoryItem] = {}

    def remember(self, item: MemoryItem) -> None:
        """Stores a validated memory item.

        Args:
            item: The MemoryItem instance to store.
        """
        self._store[item.id] = item
        logger.info("Saved memory item: %s (ID: %s, Source: %s, Importance: %s)", 
                    item.content, item.id, item.source, item.importance)

    def recall(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Retrieves and ranks relevant memories, updating their last accessed timestamp.

        Args:
            query: The search query string.
            limit: Maximum number of memories to return.

        Returns:
            list[MemoryItem]: Most relevant matching memories.
        """
        # recall uses default KEYWORD search
        results = self.search(query, search_type=SearchType.KEYWORD, limit=limit)
        
        # Update last_accessed_at for all accessed memory items
        now = datetime.now()
        for item in results:
            item.last_accessed_at = now
            # Update in-store reference
            self._store[item.id] = item

        return results

    def forget(self, item_id: str) -> bool:
        """Deletes a memory item by ID.

        Args:
            item_id: Unique string ID of memory.

        Returns:
            bool: True if item was found and deleted, False otherwise.
        """
        if item_id in self._store:
            del self._store[item_id]
            logger.info("Forgotten memory item: ID %s", item_id)
            return True
        return False

    def search(self, query: str, search_type: SearchType = SearchType.KEYWORD, limit: int = 5) -> list[MemoryItem]:
        """Searches memory items using keyword or semantic search.

        Args:
            query: The query text to search for.
            search_type: Search mode, a SearchType enum value (KEYWORD, SEMANTIC, HYBRID).
            limit: Maximum number of items to return.

        Returns:
            list[MemoryItem]: List of matching MemoryItem objects.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip().lower()

        if search_type in (SearchType.SEMANTIC, SearchType.HYBRID):
            # TODO: Integrate ChromaDB vector search in V3.0 for semantic/hybrid matches.
            # 1. Initialize chromadb client and get/create a 'memories' collection.
            # 2. Generate embeddings for queries using sentence-transformers or Gemini Embeddings API.
            # 3. Query chromadb index using embeddings and metadata filters.
            # 4. Map returned document contents back to MemoryItem models.
            logger.warning("Semantic/Hybrid search is not yet implemented (ChromaDB placeholder). Falling back to empty list.")
            return []

        # Default Keyword Search
        query_words = set(clean_query.split())
        scored_items: list[tuple[float, MemoryItem]] = []

        def _word_match(w1: str, w2: str) -> bool:
            w1_clean = w1.strip("?.!,;:\"'").lower()
            w2_clean = w2.strip("?.!,;:\"'").lower()
            if not w1_clean or not w2_clean:
                return False
            if w1_clean == w2_clean:
                return True
            # Prefix stemming for words with length >= 4 (e.g. study and studies both map to stud)
            if len(w1_clean) >= 4 and len(w2_clean) >= 4:
                return w1_clean[:4] == w2_clean[:4]
            return False

        for item in self._store.values():
            # Filter out deleted and expired items
            if getattr(item, "is_deleted", False):
                continue
            if getattr(item, "expires_at", None) and item.expires_at < datetime.now():
                continue

            item_content_lower = item.content.lower()
            item_words = set(item_content_lower.split())
            
            # Simple token overlap score with prefix/stem matching
            overlap_count = 0
            for qw in query_words:
                for iw in item_words:
                    if _word_match(qw, iw):
                        overlap_count += 1
                        break
            
            # Exact substring match bonus
            substring_bonus = 0.0
            if clean_query in item_content_lower:
                substring_bonus = 5.0

            # Factor in importance score slightly for keyword ranking (e.g. up to +0.5 bonus)
            importance_bonus = min(item.importance * 0.1, 0.5)

            match_score = overlap_count + substring_bonus
            if match_score > 0:
                score = match_score + importance_bonus
                scored_items.append((score, item))

        # Sort by score descending, then by last_accessed_at descending
        scored_items.sort(key=lambda x: (x[0], x[1].last_accessed_at), reverse=True)

        return [item for _, item in scored_items[:limit]]

    def get_all(self) -> list[MemoryItem]:
        """Retrieves all stored memory items.

        Returns:
            list[MemoryItem]: List of all saved memory items.
        """
        return list(self._store.values())

    def clear(self) -> None:
        """Clears all stored memories."""
        self._store.clear()
        logger.info("Cleared all memory storage.")
