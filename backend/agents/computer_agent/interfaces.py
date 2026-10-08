"""
Computer Interaction Agent - Interfaces.
Defines base contracts for Perception, Reasoning, Execution, Verification, and Memory.
"""

from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Tuple

from backend.agents.computer_agent.models import (
    AgentGoal,
    ComputerAction,
    PerceivedElement,
    ReasoningState,
    WorldState,
)


class BasePerception(ABC):
    """Responsible for screen capturing and translating visual layout into elements."""

    @abstractmethod
    def perceive_screen(self, monitor_index: int = 1) -> Tuple[Image.Image, WorldState, str]:
        """Captures the current screen and identifies key UI elements and description.

        Returns:
            Tuple of (PIL Image, WorldState, screen text description)
        """
        pass


class BaseReasoning(ABC):
    """Handles high-level decision making to formulate plan steps."""

    @abstractmethod
    def decide_actions(
        self,
        goal: AgentGoal,
        world_state: WorldState,
        visual_description: str,
        history: List[str],
    ) -> ReasoningState:
        """Formulates the next sequence of operations based on current perception."""
        pass


class BaseExecutor(ABC):
    """Executes actions via DesktopService."""

    @abstractmethod
    def execute_action(self, action: ComputerAction) -> Tuple[bool, str]:
        """Runs the keyboard/mouse actions. Returns (success, output message)."""
        pass


class BaseVerifier(ABC):
    """Verifies changes in screen state post-action."""

    @abstractmethod
    def verify_action(
        self,
        action: ComputerAction,
        previous_image: Image.Image,
        current_image: Image.Image,
        current_world_state: WorldState,
        current_description: str,
    ) -> Tuple[bool, str]:
        """Evaluates whether the action was successful by inspecting screen transitions."""
        pass


class BaseMemory(ABC):
    """Retains historical action logs and screenshots."""

    @abstractmethod
    def add_step(self, action_desc: str, screen_hash: str) -> None:
        """Logs an action execution and the resulting screen hash."""
        pass

    @abstractmethod
    def get_history(self) -> List[str]:
        """Returns list of human readable actions completed."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Resets the history tracking."""
        pass
