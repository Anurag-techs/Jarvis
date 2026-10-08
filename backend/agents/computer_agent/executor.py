"""
Computer Interaction Agent - Executor Component.
Executes ComputerActions by mapping them to low-level DesktopService commands.
"""

import logging
import time
from typing import Tuple

from backend.agents.computer_agent.interfaces import BaseExecutor
from backend.agents.computer_agent.models import ComputerAction, ComputerActionType
from backend.services.desktop import DesktopService

logger = logging.getLogger("jarvis.agents.computer_agent.executor")


class ComputerAgentExecutor(BaseExecutor):
    """Executes actions on the operating system via DesktopService abstraction."""

    def __init__(self, desktop_service: DesktopService) -> None:
        self._desktop = desktop_service

    def execute_action(self, action: ComputerAction) -> Tuple[bool, str]:
        logger.info("Executing action: %s (%s)", action.action_type, action.description)
        try:
            if action.action_type == ComputerActionType.MOVE_TO:
                if action.x is None or action.y is None:
                    return False, "MOVE_TO action requires x and y coordinates."
                return self._desktop.move_mouse(action.x, action.y)

            elif action.action_type == ComputerActionType.CLICK:
                if action.x is not None and action.y is not None:
                    suc, msg = self._desktop.move_mouse(action.x, action.y)
                    if not suc:
                        return False, f"Failed to move mouse before click: {msg}"
                return self._desktop.left_click()

            elif action.action_type == ComputerActionType.DOUBLE_CLICK:
                if action.x is not None and action.y is not None:
                    suc, msg = self._desktop.move_mouse(action.x, action.y)
                    if not suc:
                        return False, f"Failed to move mouse before double click: {msg}"
                return self._desktop.double_click()

            elif action.action_type == ComputerActionType.RIGHT_CLICK:
                if action.x is not None and action.y is not None:
                    suc, msg = self._desktop.move_mouse(action.x, action.y)
                    if not suc:
                        return False, f"Failed to move mouse before right click: {msg}"
                return self._desktop.right_click()

            elif action.action_type == ComputerActionType.DRAG:
                if action.x is None or action.y is None:
                    return False, "DRAG action requires target x and y coordinates."
                return self._desktop.drag(action.x, action.y)

            elif action.action_type == ComputerActionType.SCROLL:
                if action.clicks is None:
                    return False, "SCROLL action requires click count."
                return self._desktop.scroll(action.clicks)

            elif action.action_type == ComputerActionType.TYPE:
                if action.text is None:
                    return False, "TYPE action requires text parameter."
                return self._desktop.type_text(action.text)

            elif action.action_type == ComputerActionType.PRESS_KEY:
                if action.key is None:
                    return False, "PRESS_KEY action requires key parameter."
                return self._desktop.press_key(action.key)

            elif action.action_type == ComputerActionType.HOTKEY:
                if not action.keys:
                    return False, "HOTKEY action requires keys parameter."
                return self._desktop.hotkey(*action.keys)

            elif action.action_type == ComputerActionType.KEY_DOWN:
                if action.key is None:
                    return False, "KEY_DOWN action requires key parameter."
                return self._desktop.key_down(action.key)

            elif action.action_type == ComputerActionType.KEY_UP:
                if action.key is None:
                    return False, "KEY_UP action requires key parameter."
                return self._desktop.key_up(action.key)

            elif action.action_type == ComputerActionType.WAIT:
                duration = action.duration if action.duration else 1.0
                time.sleep(duration)
                return True, f"Waited for {duration} seconds."

            else:
                return False, f"Unsupported action type: {action.action_type}"

        except Exception as e:
            logger.error("Execution failed for action %s: %s", action.action_type, e)
            return False, f"Failed to execute action: {e}"
