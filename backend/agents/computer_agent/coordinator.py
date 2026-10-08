"""
Computer Interaction Agent - Coordinator (Main Orchestration Loop v2.0).
Coordinates Perception, Reasoning, Execution, Verification, Memory, Wait Conditions, and Approval Gates.
"""

import hashlib
import logging
import time
from typing import Callable, Optional, Tuple

from backend.agents.computer_agent.interfaces import (
    BasePerception,
    BaseReasoning,
    BaseExecutor,
    BaseVerifier,
    BaseMemory,
)
from backend.agents.computer_agent.models import AgentGoal, ComputerAction, ApprovalLevel

logger = logging.getLogger("jarvis.agents.computer_agent.coordinator")


class ComputerAgent:
    """Autonomous Computer Interaction Agent operating mouse/keyboard via visual loop."""

    def __init__(
        self,
        perception: BasePerception,
        reasoning: BaseReasoning,
        executor: BaseExecutor,
        verifier: BaseVerifier,
        memory: BaseMemory,
        max_retries_per_step: int = 3,
        approval_handler: Optional[Callable[[ComputerAction], bool]] = None,
    ) -> None:
        self.perception = perception
        self.reasoning = reasoning
        self.executor = executor
        self.verifier = verifier
        self.memory = memory
        self.max_retries_per_step = max_retries_per_step
        self.approval_handler = approval_handler or self._default_approval_handler

    def _default_approval_handler(self, action: ComputerAction) -> bool:
        """Default gate that allows safe and caution actions, but rejects dangerous ones unless mocked."""
        if action.approval_level == ApprovalLevel.DANGEROUS:
            logger.warning("Dangerous action '%s' blocked by default approval handler.", action.description)
            return False
        return True

    def execute_goal(self, goal_desc: str, max_steps: int = 15) -> Tuple[bool, str]:
        """Runs the continuous Observe -> Reason -> Execute -> Verify autonomous loop."""
        logger.info("Starting Autonomous ComputerAgent execution for goal: '%s'", goal_desc)
        goal = AgentGoal(description=goal_desc, max_steps=max_steps)
        
        self.memory.clear()
        # Set task memory properties
        if hasattr(self.memory, "goal"):
            self.memory.goal = goal_desc
        if hasattr(self.memory, "start_time"):
            self.memory.start_time = time.time()

        for step in range(max_steps):
            logger.info("--- Step %d of %d ---", step + 1, max_steps)

            # 1. Observe (Perception Sweep before decision)
            prev_image, world_state, prev_desc = self.perception.perceive_screen()
            
            # Update memory active window state
            if hasattr(self.memory, "current_window"):
                self.memory.current_window = world_state.active_window

            # Compute screen hash for memory logging
            import numpy as np
            resized = prev_image.resize((32, 32)).convert("L")
            screen_hash = hashlib.md5(resized.tobytes()).hexdigest()

            # 2. Reason (Uses WorldState)
            history = self.memory.get_history()
            reasoning_state = self.reasoning.decide_actions(
                goal=goal,
                world_state=world_state,
                visual_description=prev_desc,
                history=history,
            )

            # Check if goal is accomplished or terminal state reached
            if reasoning_state.is_terminal:
                logger.info("Goal execution ended by reasoning state: %s", reasoning_state.message)
                return reasoning_state.success, reasoning_state.message

            if not reasoning_state.next_actions:
                logger.warning("No actions formulated by reasoning. Terminating.")
                return False, "Reasoning engine generated empty plan."

            # 3. Execute & Verify
            for action in reasoning_state.next_actions:
                # Approval Gate Check
                if action.approval_level in (ApprovalLevel.CAUTION, ApprovalLevel.DANGEROUS):
                    logger.info("Action requires approval (%s): %s", action.approval_level, action.description)
                    approved = self.approval_handler(action)
                    if not approved:
                        logger.warning("Action execution rejected by approval gate: %s", action.description)
                        # Log rejection to memory so reasoning knows to pivot / alternative strategy
                        self.memory.add_step(f"Rejected action by approval: {action.description}", screen_hash)
                        break  # Stop sequence, return to next loop (reasoning)

                # Record actions in memory DTO
                if "click" in action.action_type.lower() and hasattr(self.memory, "last_clicked_element"):
                    self.memory.last_clicked_element = action.description
                if action.text and hasattr(self.memory, "last_typed_text"):
                    self.memory.last_typed_text = action.text

                # Execute and verify
                success = self._execute_and_verify_with_retries(
                    action=action,
                    prev_image=prev_image,
                    screen_hash=screen_hash,
                )

                if not success:
                    logger.warning("Action execution and retries failed: %s", action.description)
                    # Log failure to memory to ask planner for alternative strategy
                    self.memory.add_step(f"Failed to execute action: {action.description}", screen_hash)
                    break  # Break out of action chain to trigger fresh observation & reasoning replanning!

        return False, "ComputerAgent exceeded maximum execution steps."

    def _execute_and_verify_with_retries(
        self,
        action: ComputerAction,
        prev_image,
        screen_hash: str,
    ) -> bool:
        """Executes action and runs the verification loop, retrying on failure."""
        for attempt in range(self.max_retries_per_step):
            logger.info("Attempt %d of %d: %s", attempt + 1, self.max_retries_per_step, action.description)
            
            # 1. Execute
            exec_ok, exec_msg = self.executor.execute_action(action)
            if not exec_ok:
                logger.warning("Executor failed: %s. Retrying...", exec_msg)
                if hasattr(self.memory, "retry_count"):
                    self.memory.retry_count += 1
                time.sleep(0.5)
                continue

            # 2. Wait Condition check
            if action.wait_condition:
                logger.info("Action requires wait condition: '%s' (timeout: %.1fs)", action.wait_condition, action.wait_timeout)
                start_wait = time.time()
                condition_met = False
                while time.time() - start_wait < action.wait_timeout:
                    time.sleep(0.5)
                    _, temp_state, _ = self.perception.perceive_screen()
                    cond_lower = action.wait_condition.lower()
                    in_window = cond_lower in temp_state.active_window.lower()
                    in_elements = any(cond_lower in e.label.lower() for e in temp_state.visible_elements)
                    in_ocr = cond_lower in temp_state.ocr_text.lower()
                    
                    if in_window or in_elements or in_ocr:
                        condition_met = True
                        logger.info("Wait condition '%s' satisfied.", action.wait_condition)
                        break
                if not condition_met:
                    logger.warning("Wait condition '%s' timed out.", action.wait_condition)
                    # Proceed to verify anyway or retry

            # Pause briefly to allow state transition
            time.sleep(0.5)

            # 3. Observe post-action state
            curr_image, curr_state, curr_desc = self.perception.perceive_screen()

            # 4. Verify state transition
            verify_ok, verify_msg = self.verifier.verify_action(
                action=action,
                previous_image=prev_image,
                current_image=curr_image,
                current_world_state=curr_state,
                current_description=curr_desc,
            )

            if verify_ok:
                logger.info("Action successfully verified: %s", verify_msg)
                # Log step to memory
                self.memory.add_step(action.description, screen_hash)
                return True
            else:
                logger.warning("Verification failed: %s. Retrying...", verify_msg)
                if hasattr(self.memory, "retry_count"):
                    self.memory.retry_count += 1
                time.sleep(0.5)

        return False
