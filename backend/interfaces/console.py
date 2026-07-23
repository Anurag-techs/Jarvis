"""
JARVIS Interactive Console Interface.

1. Why this module exists:
   Fulfills Sprint 2.2 requirements by establishing an interactive keyboard input loop.

2. How it fits into the architecture:
   Presentation / Interface layer. Communicates ONLY with `SystemOrchestrator` via Dependency Injection.
   Has zero direct dependency on TTSService or underlying AI providers.

3. Which future modules will interact with it:
   - `backend.core.startup.JarvisApplication`
   - `backend.main`

4. Common mistakes to avoid:
   - Calling `TTSService` directly inside `ConsoleInterface` (orchestrator handles speech output).
   - Adding business logic or tool handling directly inside console loop.

5. Possible future improvements:
   - Rich terminal formatting via `rich` or `prompt_toolkit`.
"""

import logging
import sys

from backend.core.lifecycle import BaseLifecycleManager
from backend.core.models import AssistantResponse
from backend.core.orchestrator import SystemOrchestrator

logger = logging.getLogger("jarvis.interfaces.console")


class ConsoleInterface(BaseLifecycleManager):
    """Interactive command-line interface communicating strictly with SystemOrchestrator."""

    def __init__(self, orchestrator: SystemOrchestrator) -> None:
        """Initialize ConsoleInterface via dependency injection.

        Args:
            orchestrator: Injected SystemOrchestrator instance.
        """
        self._orchestrator = orchestrator
        self._running = False

    def start(self) -> None:
        """Starts the interactive console command loop."""
        self._running = True
        logger.info("Starting ConsoleInterface event loop")

        try:
            while self._running:
                user_input = input("You > ").strip()
                if not user_input:
                    continue

                lowered = user_input.lower()

                # Built-in Exit Commands
                if lowered in ["exit", "quit"]:
                    print("\nJARVIS: Goodbye!")
                    self.stop()
                    break

                # Built-in Developer Help Command
                if lowered in ["help", "?"]:
                    self._display_help()
                    continue

                # Delegate command processing exclusively to SystemOrchestrator
                response: AssistantResponse = self._orchestrator.process(user_input)
                print(f"\nJARVIS: {response.text}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nJARVIS: Goodbye!")
            self.stop()

    def stop(self) -> None:
        """Stops the console event loop cleanly."""
        self._running = False
        logger.info("ConsoleInterface stopped")

    def _display_help(self) -> None:
        """Displays built-in developer help documentation."""
        help_text = """
====================================================
  JARVIS Console Interface - Built-in Commands
====================================================
  help, ?      : Displays this help message.
  exit, quit   : Exits the JARVIS application cleanly.

  Sample Conversation Commands:
    - Hello
    - What is your name?
    - What time is it?
    - What is the weather in London?
====================================================
"""
        print(help_text)
