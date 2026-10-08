"""
Unit/Integration tests verifying startup bootstrapping configurations,
memory store resolution, and shutdown propagation.
"""

import os
import unittest
from unittest.mock import patch

from backend.config.settings import Settings
from backend.core.startup import StartupManager
from backend.memory.repository import SQLiteMemoryRepository


class TestMemoryLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "test_lifecycle_temp.db"
        self._cleanup()

    def tearDown(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    @patch("backend.config.settings.get_settings")
    def test_startup_shutdown_propagation(self, mock_get_settings) -> None:

        """Verifies that StartupManager configures SQLiteMemoryRepository, and app.shutdown propagates cleanly."""
        # 1. Setup mock Settings to use mock providers and our temporary DB file
        test_settings = Settings()
        test_settings.memory_db_path = self.db_path
        test_settings.stt_provider = "mock"
        test_settings.voice_provider = "mock"
        test_settings.wakeword_provider = "mock"
        test_settings.llm_provider = "mock"
        mock_get_settings.return_value = test_settings

        # 2. Bootstrap application
        manager = StartupManager()
        app = manager.bootstrap()

        # 3. Verify orchestrator configuration and dependency graph injection
        orchestrator = app.orchestrator
        self.assertIsNotNone(orchestrator)
        
        store = orchestrator._memory

        self.assertIsNotNone(store)
        
        pipeline = store.pipeline
        self.assertIsNotNone(pipeline)
        
        memory_manager = pipeline.manager
        self.assertIsNotNone(memory_manager)
        
        repository = memory_manager._repository
        self.assertIsInstance(repository, SQLiteMemoryRepository)
        self.assertEqual(repository.db_path, self.db_path)

        # The DB file should be created automatically during initialization
        self.assertTrue(os.path.exists(self.db_path), "SQLite database file was not created on startup.")

        # 4. Execute application shutdown
        app.shutdown()

        # 5. Verify database connection is closed cleanly
        self.assertIsNone(repository._conn, "SQLite database connection was not closed during shutdown.")


if __name__ == "__main__":
    unittest.main()
