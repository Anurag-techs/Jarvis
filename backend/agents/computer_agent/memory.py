"""
Computer Interaction Agent - Memory Component (v2.0).
Maintains comprehensive task memory including completed actions, focus window, retries, and elapsed time.
"""

import logging
import time
from typing import List, Optional

from backend.agents.computer_agent.interfaces import BaseMemory

logger = logging.getLogger("jarvis.agents.computer_agent.memory")


class ComputerAgentMemory(BaseMemory):
    """Maintains task state tracking and performance metrics for the autonomous session."""

    def __init__(self) -> None:
        self._history: List[str] = []
        self._screen_hashes: List[str] = []

        # v2.0 memory attributes
        self.goal: str = ""
        self.completed_actions: List[str] = []
        self.current_window: str = "Desktop"
        self.last_clicked_element: Optional[str] = None
        self.last_typed_text: Optional[str] = None
        self.retry_count: int = 0
        self.start_time: float = 0.0

    def add_step(self, action_desc: str, screen_hash: str) -> None:
        logger.info("Memory logged: '%s' | Screen Hash: %s", action_desc, screen_hash)
        self._history.append(action_desc)
        self._screen_hashes.append(screen_hash)
        self.completed_actions.append(action_desc)

    def get_history(self) -> List[str]:
        return list(self._history)

    def get_elapsed_time(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        return time.time() - self.start_time

    def clear(self) -> None:
        logger.info("Clearing agent interaction memory.")
        self._history.clear()
        self._screen_hashes.clear()

        # Reset v2.0 fields
        self.goal = ""
        self.completed_actions.clear()
        self.current_window = "Desktop"
        self.last_clicked_element = None
        self.last_typed_text = None
        self.retry_count = 0
        self.start_time = time.time()
