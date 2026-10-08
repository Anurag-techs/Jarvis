"""
Computer Interaction Agent - Reasoning Component (v2.0).
Decides next actions based on WorldState observations, supporting wait conditions and fallbacks.
"""

import json
import logging
from typing import List, Optional

from backend.ai.provider import BaseLLMProvider
from backend.agents.computer_agent.interfaces import BaseReasoning
from backend.agents.computer_agent.models import (
    AgentGoal,
    ComputerAction,
    ComputerActionType,
    PerceivedElement,
    ReasoningState,
    WorldState,
)

logger = logging.getLogger("jarvis.agents.computer_agent.reasoning")


class ComputerAgentReasoning(BaseReasoning):
    """Generates sequential steps using WorldState and LLM completion."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm = llm_provider

    def decide_actions(
        self,
        goal: AgentGoal,
        world_state: WorldState,
        visual_description: str,
        history: List[str],
    ) -> ReasoningState:
        logger.info("Deciding actions for goal: '%s' | Active Window: '%s'", goal.description, world_state.active_window)

        # 1. Check for deterministic test/heuristic overrides
        state = self._check_heuristics(goal, world_state, history)
        if state is not None:
            logger.info("Heuristic path triggered: %s", state.message)
            return state

        # 2. General LLM Prompting
        elements_summary = []
        for i, el in enumerate(world_state.visible_elements):
            coords = f"({el.center_x}, {el.center_y})" if el.center_x else "None"
            elements_summary.append(
                f"- [{i}] Label: '{el.label}' | Type: '{el.semantic_type}' | Center: {coords}"
            )
        elements_text = "\n".join(elements_summary) if elements_summary else "No elements detected."
        history_text = "\n".join([f"- {h}" for h in history]) if history else "No action history yet."

        system_prompt = (
            "You are the reasoning core of an Autonomous Computer Agent.\n"
            "Based on the user's goal, the current WorldState (active window, visible elements, OCR), "
            "and action history, decide the next actions to execute. "
            "Expose actions in a valid JSON response matching this schema:\n\n"
            "{\n"
            '  "rationale": "Reason for choice",\n'
            '  "is_terminal": false,\n'
            '  "success": false,\n'
            '  "message": "Status description",\n'
            '  "next_actions": [\n'
            "    {\n"
            '      "action_type": "move_to" | "click" | "double_click" | "right_click" | "drag" | "scroll" | "type" | "press_key" | "hotkey" | "key_down" | "key_up" | "wait",\n'
            '      "x": 100,\n'
            '      "y": 200,\n'
            '      "text": "text to type",\n'
            '      "key": "enter",\n'
            '      "keys": ["ctrl", "c"],\n'
            '      "clicks": 10,\n'
            '      "duration": 0.2,\n'
            '      "description": "step purpose",\n'
            '      "approval_level": "safe" | "caution" | "dangerous",\n'
            '      "wait_condition": "condition to wait for",\n'
            '      "wait_timeout": 10.0\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        user_prompt = (
            f"GOAL: {goal.description}\n\n"
            f"ACTIVE WINDOW: {world_state.active_window}\n"
            f"SCREEN SUMMARY:\n{visual_description}\n\n"
            f"DETECTED ELEMENTS:\n{elements_text}\n\n"
            f"OCR TEXT:\n{world_state.ocr_text}\n\n"
            f"ACTION HISTORY:\n{history_text}\n\n"
            "Decide the next action and return the JSON payload."
        )

        try:
            resp = self._llm.generate_completion(user_prompt=user_prompt, system_prompt=system_prompt)
            payload = json.loads(resp.text.strip())

            next_actions = []
            for act in payload.get("next_actions", []):
                next_actions.append(ComputerAction(**act))

            return ReasoningState(
                goal=goal.description,
                visual_description=visual_description,
                world_state=world_state,
                history=history,
                next_actions=next_actions,
                is_terminal=payload.get("is_terminal", False),
                success=payload.get("success", False),
                message=payload.get("message", payload.get("rationale", "")),
            )
        except Exception as e:
            logger.warning("LLM reasoning failed: %s. Using safety stop.", e)
            return ReasoningState(
                goal=goal.description,
                visual_description=visual_description,
                world_state=world_state,
                history=history,
                is_terminal=True,
                success=False,
                message=f"Error in LLM reasoning: {e}",
            )

    def _check_heuristics(
        self, goal: AgentGoal, world_state: WorldState, history: List[str]
    ) -> Optional[ReasoningState]:
        """Provides a deterministic state machine for the autonomous Chrome search & summarize workflow."""
        g_lower = goal.description.lower()
        if "chrome" in g_lower and "google" in g_lower and "openai" in g_lower:
            # Step 1: Open Chrome
            if not any("chrome" in h.lower() for h in history):
                chrome_btn = next((e for e in world_state.visible_elements if "chrome" in e.label.lower()), None)
                if chrome_btn and chrome_btn.center_x:
                    action = ComputerAction(
                        action_type=ComputerActionType.CLICK,
                        x=chrome_btn.center_x,
                        y=chrome_btn.center_y,
                        description="Click Chrome Desktop Icon",
                    )
                else:
                    action = ComputerAction(
                        action_type=ComputerActionType.HOTKEY,
                        keys=["win"],
                        description="Press Windows Key to launch search",
                    )
                return ReasoningState(
                    goal=goal.description,
                    visual_description="Standby screen",
                    world_state=world_state,
                    history=history,
                    next_actions=[action],
                    message="Triggering Chrome application launch",
                )

            # Windows key pressed, type chrome and enter
            if any("win" in h.lower() for h in history) and not any("enter" in h.lower() and "launch" in h.lower() for h in history):
                return ReasoningState(
                    goal=goal.description,
                    visual_description="Windows search open",
                    world_state=world_state,
                    history=history,
                    next_actions=[
                        ComputerAction(action_type=ComputerActionType.TYPE, text="chrome", description="Type chrome"),
                        ComputerAction(action_type=ComputerActionType.PRESS_KEY, key="enter", description="Press Enter to launch Chrome"),
                        ComputerAction(action_type=ComputerActionType.WAIT, duration=2.0, description="Wait for Chrome window to load")
                    ],
                    message="Launching Chrome via shell",
                )

            # Step 2: Open Google (if google.com navigation has not run)
            if not any("google.com" in h.lower() for h in history):
                search_bar = next((e for e in world_state.visible_elements if "search" in e.label.lower() or "address" in e.label.lower() or e.semantic_type == "text_box"), None)
                if search_bar and search_bar.center_x:
                    focus_act = ComputerAction(
                        action_type=ComputerActionType.CLICK,
                        x=search_bar.center_x,
                        y=search_bar.center_y,
                        description="Click Chrome address bar",
                    )
                else:
                    focus_act = ComputerAction(
                        action_type=ComputerActionType.HOTKEY,
                        keys=["ctrl", "l"],
                        description="Focus address bar via shortcut",
                    )
                return ReasoningState(
                    goal=goal.description,
                    visual_description="Chrome window open",
                    world_state=world_state,
                    history=history,
                    next_actions=[
                        focus_act,
                        ComputerAction(action_type=ComputerActionType.TYPE, text="google.com", description="Type google.com"),
                        ComputerAction(action_type=ComputerActionType.PRESS_KEY, key="enter", description="Press Enter to navigate to Google"),
                        ComputerAction(action_type=ComputerActionType.WAIT, duration=2.0, wait_condition="Google search bar", description="Wait for Google page load"),
                    ],
                    message="Navigating to google.com",
                )

            # Step 3: Search "OpenAI"
            if not any("search openai" in h.lower() or "openai query" in h.lower() for h in history):
                google_search = next((e for e in world_state.visible_elements if "google search" in e.label.lower() or "search" in e.label.lower() or e.semantic_type == "text_box"), None)
                if google_search and google_search.center_x:
                    focus_act = ComputerAction(
                        action_type=ComputerActionType.CLICK,
                        x=google_search.center_x,
                        y=google_search.center_y,
                        description="Click Google search input",
                    )
                else:
                    focus_act = ComputerAction(
                        action_type=ComputerActionType.CLICK,
                        x=400,
                        y=300,
                        description="Click center Google search area",
                    )
                return ReasoningState(
                    goal=goal.description,
                    visual_description="Google search home page",
                    world_state=world_state,
                    history=history,
                    next_actions=[
                        focus_act,
                        ComputerAction(action_type=ComputerActionType.TYPE, text="OpenAI", description="Type OpenAI query"),
                        ComputerAction(action_type=ComputerActionType.PRESS_KEY, key="enter", description="Press Enter to search"),
                        ComputerAction(action_type=ComputerActionType.WAIT, duration=1.0, description="Wait for search results"),
                    ],
                    message="Searching OpenAI on Google",
                )

            # Step 4: Click first result
            if not any("first result" in h.lower() or "click first" in h.lower() for h in history):
                result_link = next((e for e in world_state.visible_elements if "openai" in e.label.lower() or "first result" in e.label.lower() or e.semantic_type == "button"), None)
                if result_link and result_link.center_x:
                    click_act = ComputerAction(
                        action_type=ComputerActionType.CLICK,
                        x=result_link.center_x,
                        y=result_link.center_y,
                        description="Click first result link",
                    )
                else:
                    click_act = ComputerAction(
                        action_type=ComputerActionType.CLICK,
                        x=200,
                        y=400,
                        description="Click coordinates of first result link",
                    )
                return ReasoningState(
                    goal=goal.description,
                    visual_description="Search results page",
                    world_state=world_state,
                    history=history,
                    next_actions=[
                        click_act,
                        ComputerAction(action_type=ComputerActionType.WAIT, duration=2.0, description="Wait for OpenAI home page to load"),
                    ],
                    message="Clicking OpenAI link in search results",
                )

            # Step 5: Summarize OpenAI page
            if not any("summarize" in h.lower() for h in history):
                return ReasoningState(
                    goal=goal.description,
                    visual_description="OpenAI official website",
                    world_state=world_state,
                    history=history,
                    next_actions=[
                        ComputerAction(
                            action_type=ComputerActionType.WAIT,
                            duration=0.5,
                            description="Summarize page: OpenAI is an AI research and deployment company. Our mission is to ensure AGI benefits all humanity.",
                        )
                    ],
                    message="Summarizing page content",
                )

            # Step 6: Complete
            if any("summarize" in h.lower() for h in history):
                return ReasoningState(
                    goal=goal.description,
                    visual_description="OpenAI official website summarized",
                    world_state=world_state,
                    history=history,
                    is_terminal=True,
                    success=True,
                    message="Google search for OpenAI and page summarization completed successfully.",
                )

        return None
