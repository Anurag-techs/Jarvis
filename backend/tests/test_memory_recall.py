"""
Integration tests for the JARVIS Memory Recall pipeline.
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.ai.provider import MockLLMProvider
from backend.conversation.manager import ConversationManager
from backend.core.models import AssistantResponse
from backend.core.orchestrator import SystemOrchestrator
from backend.memory import MemoryPipeline, PipelineMemoryStore, MemoryRecallService
from backend.tools.registry import ToolRegistry


class TestMemoryRecall(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = MockLLMProvider(model_name="test-recall")
        self.pipeline = MemoryPipeline(llm_provider=self.llm)
        self.store = PipelineMemoryStore(pipeline=self.pipeline)
        self.recall_service = MemoryRecallService(memory_store=self.store)
        self.registry = ToolRegistry()
        
        self.conversation_manager = ConversationManager(
            llm_provider=self.llm
        )
        self.orchestrator = SystemOrchestrator(
            llm_provider=self.llm,
            tool_registry=self.registry,
            conversation_manager=self.conversation_manager,
            memory_store=self.store,
            recall_service=self.recall_service,
        )

    def test_scenario_1_user_name_recall(self) -> None:
        """Scenario 1: Store name -> recall name."""
        def custom_generate_completion(user_prompt, system_prompt=None, history=None, available_tools=None):
            lowered = user_prompt.lower()
            if "relevant memories" in lowered:
                if "anurag" in lowered and "what is my name" in lowered:
                    return AssistantResponse(text="Your name is Anurag.", tool_calls=[], success=True)
            return AssistantResponse(text="I do not have access to personal information.", tool_calls=[], success=True)

        with patch.object(self.llm, "generate_completion", side_effect=custom_generate_completion) as mock_gen:
            # First turn: User states name.
            response1 = self.orchestrator.process("my name is Anurag")
            self.assertTrue(response1.success)

            # Second turn: User asks for name
            response2 = self.orchestrator.process("What is my name?")
            self.assertTrue(response2.success)
            self.assertEqual(response2.text, "Your name is Anurag.")
            
            # Assert that the second call to generate_completion received the injected memories
            called_prompt = mock_gen.call_args_list[-1][1]["user_prompt"]
            self.assertIn("Relevant memories:", called_prompt)
            self.assertIn("User's name is Anurag.", called_prompt)
            self.assertIn("What is my name?", called_prompt)

    def test_scenario_2_user_study_recall(self) -> None:
        """Scenario 2: Store studies -> recall studies."""
        def custom_generate_completion(user_prompt, system_prompt=None, history=None, available_tools=None):
            lowered = user_prompt.lower()
            if "relevant memories" in lowered:
                if "computer science" in lowered and "what do i study" in lowered:
                    return AssistantResponse(text="You study Computer Science.", tool_calls=[], success=True)
            return AssistantResponse(text="I don't know what you study.", tool_calls=[], success=True)

        with patch.object(self.llm, "generate_completion", side_effect=custom_generate_completion) as mock_gen:
            # First turn: User states studies.
            response1 = self.orchestrator.process("i study Computer Science")
            self.assertTrue(response1.success)

            # Second turn: User asks
            response2 = self.orchestrator.process("What do I study?")
            self.assertTrue(response2.success)
            self.assertEqual(response2.text, "You study Computer Science.")

            called_prompt = mock_gen.call_args_list[-1][1]["user_prompt"]
            self.assertIn("Relevant memories:", called_prompt)
            self.assertIn("User studies Computer Science.", called_prompt)
            self.assertIn("What do I study?", called_prompt)

    def test_scenario_3_normal_conversation_no_recall(self) -> None:
        """Scenario 3: Normal conversation doesn't return or inject any memories."""
        with patch.object(self.llm, "generate_completion", return_value=AssistantResponse(text="Paris is the capital of France.", success=True)) as mock_gen:
            response = self.orchestrator.process("What is the capital of France?")
            self.assertTrue(response.success)
            self.assertEqual(response.text, "Paris is the capital of France.")
            
            called_prompt = mock_gen.call_args_list[-1][1]["user_prompt"]
            self.assertNotIn("Relevant memories:", called_prompt)
            self.assertEqual(called_prompt, "What is the capital of France?")

    def test_scenario_4_memory_store_exception_safety(self) -> None:
        """Scenario 4: If memory store fails, conversation still succeeds."""
        with patch.object(self.store, "search_relevant_facts", side_effect=RuntimeError("Store connection failed")):
            with patch.object(self.llm, "generate_completion", return_value=AssistantResponse(text="Paris.", success=True)) as mock_gen:
                response = self.orchestrator.process("What is the capital of France?")
                self.assertTrue(response.success)
                self.assertEqual(response.text, "Paris.")
                
                called_prompt = mock_gen.call_args_list[-1][1]["user_prompt"]
                self.assertEqual(called_prompt, "What is the capital of France?")


if __name__ == "__main__":
    unittest.main()
