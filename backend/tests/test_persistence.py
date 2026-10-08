"""
Integration tests verifying SQLite memory persistence, auto-loading on startup,
immediate writes, and duplicate prevention updating.
"""

import os
import sqlite3
import unittest
from datetime import datetime, timedelta

from backend.memory.manager import MemoryManager
from backend.memory.models import MemoryItem
from backend.memory.repository import SQLiteMemoryRepository


class TestPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "test_persistence_temp.db"
        self._cleanup()

    def tearDown(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_immediate_persistence(self) -> None:
        """Verifies that memories are committed to SQLite immediately upon creation."""
        repo = SQLiteMemoryRepository(db_path=self.db_path)
        manager = MemoryManager(repository=repo)

        item = MemoryItem(content="Immediate database commit test", importance=4.5, source="assistant")
        manager.remember(item)

        # Directly query the database file using raw sqlite3 to verify immediate commit
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, content, importance, source, is_deleted FROM memories WHERE id = ?", (item.id,))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Memory item was not persisted in the database immediately.")
        self.assertEqual(row["content"], "Immediate database commit test")
        self.assertEqual(row["importance"], 4.5)
        self.assertEqual(row["source"], "assistant")
        self.assertEqual(row["is_deleted"], 0)

        manager.shutdown()

    def test_startup_autoload(self) -> None:
        """Verifies that MemoryManager automatically loads active memories from SQLite on startup."""
        # 1. Create database and write memory items
        repo1 = SQLiteMemoryRepository(db_path=self.db_path)
        manager1 = MemoryManager(repository=repo1)

        item1 = MemoryItem(content="Stored memory number one", importance=2.0)
        item2 = MemoryItem(content="Stored memory number two", importance=3.5)
        manager1.remember(item1)
        manager1.remember(item2)

        manager1.shutdown()  # Close connection cleanly

        # 2. Re-instantiate manager pointing to same DB file
        repo2 = SQLiteMemoryRepository(db_path=self.db_path)
        manager2 = MemoryManager(repository=repo2)

        # Confirm memories are loaded automatically on startup
        all_items = manager2.get_all()
        self.assertEqual(len(all_items), 2)
        contents = {item.content for item in all_items}
        self.assertIn("Stored memory number one", contents)
        self.assertIn("Stored memory number two", contents)

        manager2.shutdown()

    def test_duplicate_prevention_and_updating(self) -> None:
        """Verifies that storing a duplicate fact updates the existing record rather than creating a duplicate row."""
        repo = SQLiteMemoryRepository(db_path=self.db_path)
        manager = MemoryManager(repository=repo)

        # 1. Store initial fact
        item1 = MemoryItem(content="Sarah is my sister", importance=2.0, metadata={"initial": True})
        manager.remember(item1)
        self.assertEqual(len(manager.get_all()), 1)

        # 2. Store same fact with different ID, higher importance, and new metadata
        item2 = MemoryItem(content="Sarah is my sister", importance=4.0, metadata={"updated": True})
        manager.remember(item2)

        # Confirm only one memory exists (prevented duplicate)
        all_items = manager.get_all()
        self.assertEqual(len(all_items), 1)

        # Confirm original ID was preserved but importance and metadata were updated
        merged_item = all_items[0]
        self.assertEqual(merged_item.id, item1.id)
        self.assertEqual(merged_item.importance, 4.0)
        self.assertTrue(merged_item.metadata.get("initial"))
        self.assertTrue(merged_item.metadata.get("updated"))

        # 3. Soft-delete the memory
        manager.forget(merged_item.id)
        self.assertEqual(len(manager.get_all()), 0)

        # 4. Save the same content again (should reactivate the soft-deleted memory)
        item3 = MemoryItem(content="Sarah is my sister", importance=3.0)
        manager.remember(item3)

        self.assertEqual(len(manager.get_all()), 1)
        reactivated_item = manager.get_all()[0]
        self.assertEqual(reactivated_item.id, item1.id)
        self.assertFalse(reactivated_item.is_deleted)
        self.assertEqual(reactivated_item.importance, 3.0)

        manager.shutdown()


if __name__ == "__main__":
    unittest.main()
