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


from backend.memory.models import MemoryItem, SearchType
from backend.memory.repository import BaseMemoryRepository

logger = logging.getLogger("jarvis.memory.manager")


class MemoryManager:
    """Manages storage, retrieval, search, deletion, and cleanup of MemoryItems using database persistence."""

    def __init__(self, repository: BaseMemoryRepository | None = None) -> None:
        """Initialize memory storage manager.

        Args:
            repository: Optional custom repository to inject. If None, resolves the
                        SQLiteMemoryRepository with configured application settings.
        """
        if repository is None:
            from backend.config.settings import get_settings
            from backend.memory.repository import SQLiteMemoryRepository
            settings = get_settings()
            db_path = getattr(settings, "memory_db_path", "memory.db")
            
            # Check if running under tests (pytest or unittest) and overriding default path
            import sys
            import os
            is_test = "pytest" in sys.modules or "unittest" in sys.modules or os.environ.get("TESTING") == "true"
            if is_test and db_path == "instance/jarvis_memory.db":
                db_path = ":memory:"
                
            self._repository: BaseMemoryRepository = SQLiteMemoryRepository(db_path=db_path)
        else:
            self._repository = repository

        self._store: dict[str, MemoryItem] = {}
        self._load_memories()

    def _load_memories(self) -> None:
        """Loads non-deleted and non-expired memory items from the repository into memory cache."""
        try:
            active_items = self._repository.get_all(include_deleted=False, include_expired=False)
            self._store = {item.id: item for item in active_items}
            logger.info("Loaded %d active memories from repository on startup.", len(self._store))
        except Exception as exc:
            logger.error("Failed to auto-load memories on startup: %s", exc, exc_info=True)

    def _find_by_content(self, content: str) -> MemoryItem | None:
        """Finds an existing memory item with matching content (case-insensitive, normalized)."""
        normalized_new = content.strip().lower()
        # Check active store cache first
        for item in self._store.values():
            if item.content.strip().lower() == normalized_new:
                return item
        # Check database (including deleted/expired items)
        for item in self._repository.get_all(include_deleted=True, include_expired=True):
            if item.content.strip().lower() == normalized_new:
                return item
        return None

    def remember(self, item: MemoryItem) -> None:
        """Stores a validated memory item, updating an existing one if the content matches.

        Args:
            item: The MemoryItem instance to store.
        """
        existing_item = self._find_by_content(item.content)
        if existing_item:
            # Prevent duplicate by updating existing fact
            existing_item.importance = item.importance
            existing_item.source = item.source
            existing_item.expires_at = item.expires_at
            existing_item.is_deleted = False  # Reactivate if it was soft-deleted
            existing_item.metadata.update(item.metadata)
            existing_item.last_accessed_at = datetime.now()


            # Sync the input item's properties so the caller reflects the update
            item.id = existing_item.id
            item.created_at = existing_item.created_at
            item.last_accessed_at = existing_item.last_accessed_at
            item.importance = existing_item.importance
            item.is_deleted = existing_item.is_deleted
            item.metadata = existing_item.metadata

            self._repository.save(existing_item)
            self._store[existing_item.id] = existing_item
            logger.info("Updated existing memory item: %s (ID: %s, Source: %s, Importance: %s)",
                        existing_item.content, existing_item.id, existing_item.source, existing_item.importance)
        else:
            self._repository.save(item)
            self._store[item.id] = item
            logger.info("Saved new memory item: %s (ID: %s, Source: %s, Importance: %s)",
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
            # Update cache reference and repository record
            self._store[item.id] = item
            self._repository.save(item)

        return results

    def forget(self, item_id: str) -> bool:
        """Deletes a memory item by ID (soft delete).

        Args:
            item_id: Unique string ID of memory.

        Returns:
            bool: True if item was found and deleted, False otherwise.
        """
        success = self._repository.delete(item_id)
        if success:
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

        if search_type in (SearchType.SEMANTIC, SearchType.HYBRID):
            # TODO: Integrate ChromaDB vector search in V3.0 for semantic/hybrid matches.
            logger.warning("Semantic/Hybrid search is not yet implemented (ChromaDB placeholder). Falling back to empty list.")
            return []

        # Delegate keyword search directly to repository
        return self._repository.search(query, limit=limit)

    def get_all(self) -> list[MemoryItem]:
        """Retrieves all stored memory items from active cache.

        Returns:
            list[MemoryItem]: List of all saved memory items.
        """
        return list(self._store.values())

    def clear(self) -> None:
        """Clears all stored memories."""
        self._store.clear()
        self._repository.clear()
        logger.info("Cleared all memory storage.")

    def shutdown(self) -> None:
        """Cleanly releases any open connection resources in the repository."""
        logger.info("Shutting down MemoryManager and releasing repository connections...")
        self._repository.close()

