"""
JARVIS Memory Repository Abstraction and Implementations.

1. Why this module exists:
   Defines the data access contract (BaseMemoryRepository) for saving, retrieving,
   soft-deleting, and searching memories, and provides concrete SQLite and In-Memory implementations.
   This facilitates swapping database layers (e.g. SQLite to ChromaDB or vector store) without affecting orchestrators.

2. How it fits into the architecture:
   Sits in the Memory abstraction layer, decoupling MemoryManager from raw SQLite commands.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any

from backend.memory.models import MemoryItem

logger = logging.getLogger("jarvis.memory.repository")


def _format_datetime(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_datetime(dt_str: str | None) -> datetime | None:
    if dt_str is None:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        return datetime.now()


def keyword_search(items: list[MemoryItem], query: str, limit: int = 5) -> list[MemoryItem]:
    """Helper to perform the keyword search and scoring algorithm across a list of MemoryItems.

    Matches words (with prefix-stemming of length >= 4) and awards a substring match bonus and
    slight importance bonus.
    """
    if not query or not query.strip():
        return []

    clean_query = query.strip().lower()
    query_words = set(clean_query.split())
    scored_items: list[tuple[float, MemoryItem]] = []

    def _word_match(w1: str, w2: str) -> bool:
        w1_clean = w1.strip("?.!,;:\"'").lower()
        w2_clean = w2.strip("?.!,;:\"'").lower()
        if not w1_clean or not w2_clean:
            return False
        if w1_clean == w2_clean:
            return True
        if len(w1_clean) >= 4 and len(w2_clean) >= 4:
            return w1_clean[:4] == w2_clean[:4]
        return False

    for item in items:
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


class BaseMemoryRepository(ABC):
    """Abstract Base Class defining the contract for memory storage repositories."""

    @abstractmethod
    def save(self, item: MemoryItem) -> None:
        """Saves a MemoryItem to the store. Inserts if new, updates existing fields on conflict."""
        pass

    @abstractmethod
    def get_by_id(self, item_id: str) -> MemoryItem | None:
        """Retrieves a MemoryItem by its unique ID."""
        pass

    @abstractmethod
    def get_all(self, include_deleted: bool = False, include_expired: bool = False) -> list[MemoryItem]:
        """Retrieves all memory items from the store, with filters for deleted and expired items."""
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Marks a memory item as deleted (soft delete). Returns True if found and deleted."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Searches the repository for memories matching the query."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all stored memories."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes any open resources (e.g. database connections)."""
        pass


class InMemoryMemoryRepository(BaseMemoryRepository):
    """Fallback/Test memory repository that stores items in-memory."""

    def __init__(self) -> None:
        self._store: dict[str, MemoryItem] = {}

    def save(self, item: MemoryItem) -> None:
        self._store[item.id] = item

    def get_by_id(self, item_id: str) -> MemoryItem | None:
        return self._store.get(item_id)

    def get_all(self, include_deleted: bool = False, include_expired: bool = False) -> list[MemoryItem]:
        now = datetime.now()
        results = []
        for item in self._store.values():
            if not include_deleted and item.is_deleted:
                continue
            if not include_expired and item.expires_at and item.expires_at < now:
                continue
            results.append(item)
        return results

    def delete(self, item_id: str) -> bool:
        if item_id in self._store:
            self._store[item_id].is_deleted = True
            return True
        return False

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        active_items = self.get_all(include_deleted=False, include_expired=False)
        return keyword_search(active_items, query, limit=limit)

    def clear(self) -> None:
        self._store.clear()

    def close(self) -> None:
        pass


class SQLiteMemoryRepository(BaseMemoryRepository):
    """Production memory repository that persists items in a SQLite database."""

    def __init__(self, db_path: str = "memory.db") -> None:
        """Initialize SQLite database repository connection.

        Args:
            db_path: Path to the database file (supports ':memory:' for transient sqlite stores).
        """
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                # Automatically create parent directories if missing
                db_dir = os.path.dirname(os.path.abspath(self.db_path))
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                importance REAL NOT NULL,
                source TEXT NOT NULL,
                expires_at TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.commit()
        logger.debug("SQLite database initialized at: %s", self.db_path)

    def save(self, item: MemoryItem) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO memories (id, content, created_at, last_accessed_at, importance, source, expires_at, is_deleted, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content,
                last_accessed_at=excluded.last_accessed_at,
                importance=excluded.importance,
                source=excluded.source,
                expires_at=excluded.expires_at,
                is_deleted=excluded.is_deleted,
                metadata=excluded.metadata
            """,
            (
                item.id,
                item.content,
                _format_datetime(item.created_at),
                _format_datetime(item.last_accessed_at),
                item.importance,
                item.source,
                _format_datetime(item.expires_at),
                1 if item.is_deleted else 0,
                json.dumps(item.metadata),
            ),
        )
        conn.commit()

    def get_by_id(self, item_id: str) -> MemoryItem | None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, created_at, last_accessed_at, importance, source, expires_at, is_deleted, metadata FROM memories WHERE id = ?",
            (item_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            created_at=_parse_datetime(row["created_at"]),
            last_accessed_at=_parse_datetime(row["last_accessed_at"]),
            importance=row["importance"],
            source=row["source"],
            expires_at=_parse_datetime(row["expires_at"]),
            is_deleted=bool(row["is_deleted"]),
            metadata=json.loads(row["metadata"]),
        )

    def get_all(self, include_deleted: bool = False, include_expired: bool = False) -> list[MemoryItem]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, created_at, last_accessed_at, importance, source, expires_at, is_deleted, metadata FROM memories"
        )
        rows = cursor.fetchall()

        items = []
        now = datetime.now()
        for row in rows:
            is_deleted = bool(row["is_deleted"])
            expires_at = _parse_datetime(row["expires_at"])

            if not include_deleted and is_deleted:
                continue
            if not include_expired and expires_at and expires_at < now:
                continue

            items.append(
                MemoryItem(
                    id=row["id"],
                    content=row["content"],
                    created_at=_parse_datetime(row["created_at"]),
                    last_accessed_at=_parse_datetime(row["last_accessed_at"]),
                    importance=row["importance"],
                    source=row["source"],
                    expires_at=expires_at,
                    is_deleted=is_deleted,
                    metadata=json.loads(row["metadata"]),
                )
            )
        return items

    def delete(self, item_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE memories SET is_deleted = 1 WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        active_items = self.get_all(include_deleted=False, include_expired=False)
        return keyword_search(active_items, query, limit=limit)

    def clear(self) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories")
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as e:
                logger.warning("Error closing SQLite connection: %s", e)
            finally:
                self._conn = None
