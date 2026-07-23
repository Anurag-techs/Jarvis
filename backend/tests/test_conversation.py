"""
Unit tests for JARVIS ConversationManager (Sprint 3.5).
"""

import unittest
from backend.ai.provider import MockLLMProvider
from backend.conversation.manager import ConversationManager
from backend.conversation.system_prompt import DefaultSystemPromptProvider
from backend.core.models import AssistantResponse, ConversationMessage


class TestConversationManager(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = MockLLMProvider()
        self.manager = ConversationManager(
            llm_provider=self.llm,
            history_limit=4,
            session_id="test_session_101",
        )

    def test_add_messages_and_ordering(self) -> None:
        """Verifies storing user/assistant messages and maintaining chronological order."""
        msg1 = self.manager.add_user_message("Hello JARVIS")
        msg2 = self.manager.add_assistant_message("Hello! I am JARVIS.")

        history = self.manager.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "Hello JARVIS")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].content, "Hello! I am JARVIS.")
        self.assertIsNotNone(history[0].timestamp)

    def test_clear_history(self) -> None:
        """Verifies clearing conversation history."""
        self.manager.add_user_message("Test input")
        self.manager.add_assistant_message("Test output")
        self.assertEqual(len(self.manager.get_history()), 2)

        self.manager.clear()
        self.assertEqual(len(self.manager.get_history()), 0)

    def test_history_trimming(self) -> None:
        """Verifies history is trimmed when exceeding history_limit (limit = 4)."""
        # Add 6 messages (3 user/assistant turns)
        self.manager.add_user_message("Turn 1 User")
        self.manager.add_assistant_message("Turn 1 Assistant")
        self.manager.add_user_message("Turn 2 User")
        self.manager.add_assistant_message("Turn 2 Assistant")
        self.manager.add_user_message("Turn 3 User")
        self.manager.add_assistant_message("Turn 3 Assistant")

        # Trim history
        self.manager._trim_history()
        history = self.manager.get_history()

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].content, "Turn 2 User")
        self.assertEqual(history[-1].content, "Turn 3 Assistant")

    def test_generate_response_pipeline(self) -> None:
        """Verifies generate_response adds user prompt, queries LLM, adds response, and returns AssistantResponse."""
        response = self.manager.generate_response("Hello")
        self.assertIsInstance(response, AssistantResponse)
        self.assertTrue(response.success)
        self.assertEqual(response.text, "Hello! I am JARVIS.")

        history = self.manager.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(response.metadata.get("session_id"), "test_session_101")

    def test_custom_system_prompt_provider(self) -> None:
        """Verifies custom SystemPromptProvider injection."""
        custom_provider = DefaultSystemPromptProvider(assistant_name="JARVIS-Custom")
        manager = ConversationManager(
            llm_provider=self.llm,
            system_prompt_provider=custom_provider,
        )
        response = manager.generate_response("Who are you?")
        self.assertTrue(response.success)


if __name__ == "__main__":
    unittest.main()
