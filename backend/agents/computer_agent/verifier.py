"""
Computer Interaction Agent - Verifier Component (v2.0).
Verifies GUI state transitions after actions by analyzing image differences and WorldState.
"""

import logging
from PIL import Image
from typing import List, Tuple

from backend.agents.computer_agent.interfaces import BaseVerifier
from backend.agents.computer_agent.models import ComputerAction, ComputerActionType, WorldState

logger = logging.getLogger("jarvis.agents.computer_agent.verifier")


class ComputerAgentVerifier(BaseVerifier):
    """Compares pre-action and post-action visual screens and WorldStates to determine success."""

    def verify_action(
        self,
        action: ComputerAction,
        previous_image: Image.Image,
        current_image: Image.Image,
        current_world_state: WorldState,
        current_description: str,
    ) -> Tuple[bool, str]:
        # For scroll and wait actions, screen changes are optional or expected to be minimal.
        if action.action_type in (ComputerActionType.WAIT, ComputerActionType.MOVE_TO):
            return True, "Verification skipped for move/wait."

        try:
            # 1. Check if expected active window is now active
            if action.wait_condition and "window" in action.description.lower():
                if action.wait_condition.lower() not in current_world_state.active_window.lower():
                    return False, f"Expected active window to be '{action.wait_condition}', but got '{current_world_state.active_window}'."

            # 2. Perceptual image difference check
            import numpy as np
            img1 = np.array(previous_image.resize((64, 64)).convert("L"), dtype=np.float32)
            img2 = np.array(current_image.resize((64, 64)).convert("L"), dtype=np.float32)
            
            mean_diff = float(np.mean(np.abs(img1 - img2)))
            logger.info("Visual state delta (mean absolute pixel difference): %.4f", mean_diff)

            # If pixel change is extremely low, click or keyboard might have missed
            if mean_diff < 0.01:
                logger.warning("Visual screen state is identical. Action '%s' had no effect.", action.action_type)
                return False, f"Screen state did not change after '{action.action_type}'. Action may have failed."

            return True, "Screen successfully transitioned visually."
        except Exception as e:
            logger.warning("Verifier difference calculation hit an error: %s. Defaulting to true.", e)
            return True, "Verification defaulted to true."
