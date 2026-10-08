"""
Unit tests for JARVIS Memory Repositories (InMemoryMemoryRepository and SQLiteMemoryRepository).
"""

from datetime import datetime, timedelta
import threading
import unittest

from backend.memory.models import MemoryItem
from backend.memory.repository import (
    InMemoryMemoryRepository,
    SQLiteMemoryRepository,
    keyword_search,
)


class TestMemoryRepositories(unittest.TestCase):
    def test_in_memory_repository_crud(self) -> None:
        """Verifies CRUD operations on InMemoryMemoryRepository."""
        repo = InMemoryMemoryRepository()
        item = MemoryItem(content="User prefers hot chocolate", importance=3.0, source="user")

        # Save & Get by ID
        repo.save(item)
        retrieved = repo.get_by_id(item.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "User prefers hot chocolate")

        # Get all active
        all_items = repo.get_all()
        self.assertEqual(len(all_items), 1)

        # Soft Delete
        success = repo.delete(item.id)
        self.assertTrue(success)

        # Confirm not returned in default get_all()
        self.assertEqual(len(repo.get_all(include_deleted=False)), 0)
        # Confirm returned when include_deleted is True
        self.assertEqual(len(repo.get_all(include_deleted=True)), 1)
        self.assertTrue(repo.get_all(include_deleted=True)[0].is_deleted)

    def test_sqlite_repository_crud(self) -> None:
        """Verifies CRUD operations on SQLiteMemoryRepository using an in-memory SQLite database."""
        repo = SQLiteMemoryRepository(db_path=":memory:")
        item = MemoryItem(content="Anurag resides in Bengaluru", importance=4.0, source="user")

        # Save & Get by ID
        repo.save(item)
        retrieved = repo.get_by_id(item.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Anurag resides in Bengaluru")
        self.assertEqual(retrieved.importance, 4.0)
        self.assertEqual(retrieved.source, "user")
        self.assertFalse(retrieved.is_deleted)

        # Soft Delete
        success = repo.delete(item.id)
        self.assertTrue(success)

        # Query all
        self.assertEqual(len(repo.get_all(include_deleted=False)), 0)
        self.assertEqual(len(repo.get_all(include_deleted=True)), 1)

        # Clear
        repo.clear()
        self.assertEqual(len(repo.get_all(include_deleted=True)), 0)
        repo.close()

    def test_repository_expiration(self) -> None:
        """Verifies retrieval filters expired memory items correctly."""
        repo = SQLiteMemoryRepository(db_path=":memory:")
        
        past_time = datetime.now() - timedelta(minutes=1)
        future_time = datetime.now() + timedelta(minutes=10)

        expired_item = MemoryItem(content="Expired fact", expires_at=past_time)
        valid_item = MemoryItem(content="Valid fact", expires_at=future_time)
        no_expiration_item = MemoryItem(content="Permanent fact", expires_at=None)

        repo.save(expired_item)
        repo.save(valid_item)
        repo.save(no_expiration_item)

        # Retrieve all active (should exclude expired)
        active = repo.get_all(include_expired=False)
        self.assertEqual(len(active), 2)
        contents = {item.content for item in active}
        self.assertNotIn("Expired fact", contents)

        # Retrieve all including expired
        all_items = repo.get_all(include_expired=True)
        self.assertEqual(len(all_items), 3)
        repo.close()

    def test_keyword_search_scoring(self) -> None:
        """Verifies prefix-stemming, overlap matching, and importance bonus in keyword_search."""
        item1 = MemoryItem(content="I study computer engineering", importance=5.0)
        item2 = MemoryItem(content="I dislike cold weather", importance=1.0)
        item3 = MemoryItem(content="My sister is called Sarah", importance=3.0)
        items = [item1, item2, item3]

        # Exact substring match bonus test
        results = keyword_search(items, "cold weather")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "I dislike cold weather")

        # Stemming prefix test (study vs studies/studying prefix match len>=4)
        results_stem = keyword_search(items, "studying computer")
        self.assertEqual(len(results_stem), 1)
        self.assertEqual(results_stem[0].content, "I study computer engineering")

        # Relevance order test (Sarah matches sister and Sarah)
        results_importance = keyword_search(items, "Sarah")
        self.assertEqual(len(results_importance), 1)
        self.assertEqual(results_importance[0].content, "My sister is called Sarah")

    def test_concurrent_access_sqlite(self) -> None:
        """Verifies SQLiteMemoryRepository handles concurrent database writes safely from multiple threads."""
        repo = SQLiteMemoryRepository(db_path=":memory:")
        errors = []

        def worker(index: int) -> None:
            try:
                item = MemoryItem(content=f"Concurrent memory {index}", importance=2.0)
                repo.save(item)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent database write failed: {errors}")
        self.assertEqual(len(repo.get_all()), 10)
        repo.close()


if __name__ == "__main__":
    unittest.main()
