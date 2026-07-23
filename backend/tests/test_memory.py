"""
Unit tests for the JARVIS Memory abstraction and concrete implementations.
"""

from datetime import datetime, timedelta
import unittest
from backend.ai.provider import MockLLMProvider
from backend.memory.models import ExtractedMemory, MemoryItem, SearchType
from backend.memory.pre_filter import MemoryPreFilter
from backend.memory.extractor import MemoryExtractor
from backend.memory.summarizer import MemorySummarizer
from backend.memory.manager import MemoryManager
from backend.memory.pipeline import MemoryPipeline
from backend.memory.base import PipelineMemoryStore


class TestMemoryComponents(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_llm = MockLLMProvider(model_name="test-memory")
        self.manager = MemoryManager()
        self.pre_filter = MemoryPreFilter()
        self.extractor = MemoryExtractor(llm_provider=self.mock_llm)
        self.summarizer = MemorySummarizer(llm_provider=self.mock_llm)
        self.pipeline = MemoryPipeline(
            llm_provider=self.mock_llm,
            manager=self.manager,
            pre_filter=self.pre_filter,
            extractor=self.extractor,
            summarizer=self.summarizer,
            prefilter_threshold=0.5,
        )

    def test_memory_item_defaults(self) -> None:
        """Verifies that MemoryItem initializes with generated UUID, default importance/source, and valid timestamps."""
        item = MemoryItem(content="User prefers green tea")
        self.assertIsNotNone(item.id)
        self.assertIsInstance(item.created_at, datetime)
        self.assertIsInstance(item.last_accessed_at, datetime)
        self.assertEqual(item.importance, 1.0)
        self.assertEqual(item.source, "user")
        self.assertEqual(item.metadata, {})

    def test_pre_filter_confidence_scoring(self) -> None:
        """Verifies pre-filter calculates correct confidence scores based on heuristics and keywords."""
        # Trivial short inputs should be 0.0 confidence
        self.assertEqual(self.pre_filter.calculate_confidence("yes"), 0.0)
        self.assertEqual(self.pre_filter.calculate_confidence("hello"), 0.0)
        self.assertEqual(self.pre_filter.calculate_confidence("how are you?"), 0.0)

        # Non-first person or generic long query should be 0.0 confidence
        self.assertEqual(self.pre_filter.calculate_confidence("The weather in Tokyo is very rainy today."), 0.0)

        # Keyword and first-person should combine (0.5 for keyword + 0.3 for first-person = 0.8)
        self.assertAlmostEqual(self.pre_filter.calculate_confidence("remember that my sister is Sarah"), 0.8)
        
        # Heuristics (0.3 first-person + 0.3 preference = 0.6)
        self.assertAlmostEqual(self.pre_filter.calculate_confidence("i prefer drinking black coffee in the morning"), 0.6)

    def test_mock_extractor_fields(self) -> None:
        """Verifies MockLLMProvider extraction captures importance and source fields."""
        mems = self.extractor.extract_memories("my name is Anurag", "Nice to meet you, Anurag.")
        self.assertEqual(len(mems), 1)
        self.assertIsInstance(mems[0], ExtractedMemory)
        self.assertEqual(mems[0].content, "User's name is Anurag.")
        self.assertEqual(mems[0].importance, 5.0)
        self.assertEqual(mems[0].source, "user")
        self.assertEqual(mems[0].metadata.get("category"), "user_profile")

    def test_manager_storage_lifecycle_and_enum_search(self) -> None:
        """Verifies remember, forget, clear, and enum-based retrieval in MemoryManager."""
        item1 = MemoryItem(content="Anurag likes coding in Python", importance=4.5, source="user")
        item2 = MemoryItem(content="Anurag dislikes cold weather", importance=2.0, source="user")

        # Remember
        self.manager.remember(item1)
        self.manager.remember(item2)
        self.assertEqual(len(self.manager.get_all()), 2)

        # Keyword Search using Enum
        results = self.manager.search("Python", search_type=SearchType.KEYWORD)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, item1.id)
        self.assertEqual(results[0].importance, 4.5)
        self.assertEqual(results[0].source, "user")

        # Forget
        success = self.manager.forget(item1.id)
        self.assertTrue(success)
        self.assertEqual(len(self.manager.get_all()), 1)

        # Clear
        self.manager.clear()
        self.assertEqual(len(self.manager.get_all()), 0)

    def test_manager_recall_updates_accessed_at(self) -> None:
        """Verifies that calling recall() updates last_accessed_at on matched MemoryItems."""
        past_time = datetime.now() - timedelta(hours=1)
        item = MemoryItem(content="Likes reading novels", last_accessed_at=past_time)
        self.manager.remember(item)

        # Verify initial last_accessed_at is in the past
        self.assertEqual(self.manager.get_all()[0].last_accessed_at, past_time)

        # Recall should update last_accessed_at
        recalled = self.manager.recall("reading novels")
        self.assertEqual(len(recalled), 1)
        self.assertGreater(recalled[0].last_accessed_at, past_time)

    def test_semantic_search_placeholder(self) -> None:
        """Verifies semantic and hybrid searches fallback to empty list."""
        item = MemoryItem(content="Works at Google")
        self.manager.remember(item)

        results_semantic = self.manager.search("Google", search_type=SearchType.SEMANTIC)
        self.assertEqual(results_semantic, [])

        results_hybrid = self.manager.search("Google", search_type=SearchType.HYBRID)
        self.assertEqual(results_hybrid, [])

    def test_pipeline_threshold_filtering(self) -> None:
        """Verifies pipeline filters out inputs below threshold and processes inputs above threshold."""
        # 1. High threshold (0.9) - Should skip a query with confidence 0.6
        high_threshold_pipeline = MemoryPipeline(
            llm_provider=self.mock_llm,
            manager=self.manager,
            pre_filter=self.pre_filter,
            extractor=self.extractor,
            summarizer=self.summarizer,
            prefilter_threshold=0.9
        )
        new_items = high_threshold_pipeline.process_interaction(
            "i prefer drinking black coffee", 
            "Okay."
        )
        self.assertEqual(new_items, [])
        self.assertEqual(len(self.manager.get_all()), 0)

        # 2. Low threshold (0.5) - Should process the same query (confidence 0.6)
        low_threshold_pipeline = MemoryPipeline(
            llm_provider=self.mock_llm,
            manager=self.manager,
            pre_filter=self.pre_filter,
            extractor=self.extractor,
            summarizer=self.summarizer,
            prefilter_threshold=0.5
        )
        new_items = low_threshold_pipeline.process_interaction(
            "i prefer drinking black coffee", 
            "Okay."
        )
        # Note: since MockLLM doesn't map "i prefer...", it maps "i like" or "remember that" or "my name is", 
        # let's use a query that maps in mock extractor to test full flow
        new_items = low_threshold_pipeline.process_interaction(
            "remember that i live in Mumbai", 
            "Got it."
        )
        self.assertEqual(len(new_items), 1)
        self.assertEqual(new_items[0].content, "User lives in Mumbai.")
        self.assertEqual(new_items[0].importance, 4.0)
        self.assertEqual(new_items[0].source, "user")
        self.assertEqual(len(self.manager.get_all()), 1)

    def test_pipeline_store_bridge(self) -> None:
        """Verifies that PipelineMemoryStore correctly delegates to the memory pipeline and manager."""
        store = PipelineMemoryStore(pipeline=self.pipeline)
        
        # Test store_interaction saves recent context and executes pipeline
        store.store_interaction(
            user_query="remember that my name is Anurag", 
            assistant_response="Got it.",
            metadata={"source": "test"}
        )
        
        # Check recent context
        context = store.get_recent_context(limit=1)
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["user"], "remember that my name is Anurag")

        # Check facts searched via store
        facts = store.search_relevant_facts("Anurag")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0], "User's name is Anurag.")


if __name__ == "__main__":
    unittest.main()
