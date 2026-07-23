"""
Unit tests for JARVIS ConsoleInterface (Sprint 2.2).
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.core.models import AssistantResponse
from backend.interfaces.console import ConsoleInterface


class TestConsoleInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = MagicMock()
        self.orchestrator.process.return_value = AssistantResponse(
            text="Hello! I am JARVIS.",
            should_speak=True,
            success=True,
        )
        self.console = ConsoleInterface(orchestrator=self.orchestrator)

    @patch("builtins.input", side_effect=["Hello", "exit"])
    @patch("builtins.print")
    def test_console_conversation_flow(self, mock_print, mock_input) -> None:
        """Verifies input prompt loop and orchestrator delegation."""
        self.console.start()
        self.orchestrator.process.assert_called_once_with("Hello")
        mock_print.assert_any_call("\nJARVIS: Hello! I am JARVIS.\n")

    @patch("builtins.input", side_effect=["help", "quit"])
    @patch("builtins.print")
    def test_console_help_command(self, mock_print, mock_input) -> None:
        """Verifies built-in help command displays help text without calling orchestrator."""
        self.console.start()
        self.orchestrator.process.assert_not_called()

    @patch("builtins.input", side_effect=["exit"])
    @patch("builtins.print")
    def test_console_exit_command(self, mock_print, mock_input) -> None:
        """Verifies clean exit on 'exit' command."""
        self.console.start()
        mock_print.assert_any_call("\nJARVIS: Goodbye!")


if __name__ == "__main__":
    unittest.main()
