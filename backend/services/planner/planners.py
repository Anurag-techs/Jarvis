"""
JARVIS Planners.
Concrete swappable implementations for high-level plan generation (LLM and Mock).
"""

import json
import logging
from typing import Any

from backend.ai.provider import BaseLLMProvider
from backend.services.planner.interfaces import BasePlanner
from backend.services.planner.models import PlanStep

logger = logging.getLogger("jarvis.services.planner.planners")


class LLMPlanner(BasePlanner):
    """Planner utilizing Google Gemini to decompose goals into structured steps."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        """Initialize LLM planner.

        Args:
            llm_provider: Active LLM provider.
        """
        self._llm = llm_provider

    def generate_plan(
        self,
        goal: str,
        available_tools: list[dict[str, Any]]
    ) -> list[PlanStep]:
        """Queries the Gemini LLM to decompose a goal into a list of structured steps.

        Args:
            goal: Target user goal.
            available_tools: Registered tool schemas.

        Returns:
            List of PlanStep models.
        """
        from backend.ai.provider import MockLLMProvider
        if isinstance(self._llm, MockLLMProvider):
            logger.info("MockLLMProvider active: returning default mock plan steps.")
            return MockPlanner().generate_plan(goal, available_tools)

        logger.info("Generating execution steps for goal: '%s'...", goal)
        try:
            system_instruction = (
                "You are an expert planning agent. You must analyze the user's high-level goal "
                "and decompose it into a logical, ordered sequence of steps using the available tools.\n"
                "Your steps should be fine-grained and tool-agnostic at the framework level, "
                "meaning they invoke tools dynamically by specifying the tool name and arguments.\n\n"
                "You also support step dependencies: if a step requires previous steps to finish first, "
                "list their IDs in 'depends_on'. Every step can optionally include a 'verification_criteria' "
                "string describing what visual state must exist on screen to confirm success.\n\n"
                "REQUIRED JSON OUTPUT SCHEMA:\n"
                "{\n"
                '  "steps": [\n'
                '    {\n'
                '      "id": "step_1",\n'
                '      "description": "Step description",\n'
                '      "action_type": "tool_call",\n'
                '      "parameters": {\n'
                '        "tool_name": "desktop_automation",\n'
                '        "arguments": {\n'
                '          "action": "open_browser",\n'
                '          "args": {"url": "https://www.example.com"}\n'
                '        }\n'
                '      },\n'
                '      "depends_on": [],\n'
                '      "verification_criteria": "Visual confirmation statement"\n'
                '    }\n'
                '  ]\n'
                "}"
            )

            prompt = (
                f"GOAL: {goal}\n\n"
                f"AVAILABLE TOOLS:\n{json.dumps(available_tools, indent=2)}"
            )

            response_raw = self._llm.generate_raw_completion(
                user_prompt=prompt,
                system_prompt=system_instruction,
                response_mime_type="application/json"
            )

            payload = json.loads(response_raw.strip())
            raw_steps = payload.get("steps", [])

            steps = []
            for step_data in raw_steps:
                steps.append(PlanStep.model_validate(step_data))

            logger.info("Successfully generated plan with %d steps.", len(steps))
            return steps

        except Exception as exc:
            logger.error("LLM Plan generation failed: %s", exc)
            # Return a simple fallback step so we don't return an empty plan
            return [
                PlanStep(
                    id="step_fallback",
                    description=f"Fallback single-step to execute: {goal}",
                    action_type="tool_call",
                    parameters={
                        "tool_name": "desktop_automation",
                        "arguments": {"action": "search_browser", "args": {"query": goal}}
                    }
                )
            ]


class MockPlanner(BasePlanner):
    """Mock planner returning pre-defined steps for testing purposes."""

    def generate_plan(
        self,
        goal: str,
        available_tools: list[dict[str, Any]]
    ) -> list[PlanStep]:
        """Produces a deterministic two-step mock plan."""
        logger.debug("MockPlanner generating test plan steps.")
        return [
            PlanStep(
                id="step_1",
                description="Set target text to clipboard",
                action_type="tool_call",
                parameters={
                    "tool_name": "desktop_automation",
                    "arguments": {
                        "action": "set_clipboard",
                        "args": {"text": "JARVIS PLANNER ACTIVE"}
                    }
                },
                depends_on=[],
                verification_criteria="Clipboard holds the target text"
            ),
            PlanStep(
                id="step_2",
                description="Read and verify text from clipboard",
                action_type="tool_call",
                parameters={
                    "tool_name": "desktop_automation",
                    "arguments": {"action": "get_clipboard"}
                },
                depends_on=["step_1"],
                verification_criteria="Clipboard content retrieved successfully"
            )
        ]
