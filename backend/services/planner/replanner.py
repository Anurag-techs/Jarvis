"""
JARVIS Planner Replanning Engine.
Re-decomposes and revises active plans mid-execution upon step failures.
"""

import json
import logging
from typing import Any

from backend.ai.provider import BaseLLMProvider
from backend.services.planner.models import PlanStep

logger = logging.getLogger("jarvis.services.planner.replanner")


class Replanner:
    """Invokes LLM reasoning to modify remaining plan steps when execution faults occur."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        """Initialize replanner.

        Args:
            llm_provider: Active LLM provider.
        """
        self._llm = llm_provider

    def replan(
        self,
        goal: str,
        completed_steps: list[PlanStep],
        remaining_steps: list[PlanStep],
        failed_step: PlanStep,
        error_message: str,
        available_tools: list[dict[str, Any]],
    ) -> list[PlanStep]:
        """Queries the LLM provider to formulate a recovery plan.

        Args:
            goal: Original high-level goal.
            completed_steps: List of steps already successfully executed.
            remaining_steps: List of planned steps that have not run yet.
            failed_step: The step that failed execution.
            error_message: Failure diagnostic message.
            available_tools: List of active system tool schemas.

        Returns:
            List of revised remaining PlanStep models.
        """
        from backend.ai.provider import MockLLMProvider
        if isinstance(self._llm, MockLLMProvider):
            logger.info("MockLLMProvider active: executing mock plan recovery route.")
            # For testing: return remaining steps with the failed step replaced/modified
            recovery_step = PlanStep(
                id=f"{failed_step.id}_alt",
                description=f"Alternative route for failed: {failed_step.description}",
                action_type=failed_step.action_type,
                parameters=failed_step.parameters,
                verification_criteria="Verify alternative route"
            )
            return [recovery_step] + remaining_steps

        logger.info("Executing LLM replanning for failed step %s...", failed_step.id)
        try:
            completed_desc = [f"- {s.id}: {s.description} (result={s.result})" for s in completed_steps]
            remaining_desc = [f"- {s.id}: {s.description}" for s in remaining_steps]

            system_instruction = (
                "You are an expert replanning agent. An execution plan failed mid-way. "
                "You must inspect the goal, completed steps, the failed step with its error, "
                "and the remaining planned steps. Propose a revised sequence of remaining steps "
                "to achieve the goal using the available tools.\n\n"
                "Output your response strictly as JSON matching this schema:\n"
                "{\n"
                '  "revised_steps": [\n'
                '    {\n'
                '      "id": "step_id",\n'
                '      "description": "Step description",\n'
                '      "action_type": "tool_call or wait",\n'
                '      "parameters": {"tool_name": "...", "arguments": {...}},\n'
                '      "depends_on": [],\n'
                '      "verification_criteria": "Optional criteria"\n'
                '    }\n'
                '  ]\n'
                "}"
            )

            prompt = (
                f"GOAL: {goal}\n\n"
                f"COMPLETED STEPS:\n" + ("\n".join(completed_desc) if completed_desc else "(none)") + "\n\n"
                f"FAILED STEP: {failed_step.id}: {failed_step.description}\n"
                f"ERROR: {error_message}\n\n"
                f"REMAINING PLANNED STEPS:\n" + ("\n".join(remaining_desc) if remaining_desc else "(none)") + "\n\n"
                f"AVAILABLE TOOLS:\n{json.dumps(available_tools, indent=2)}"
            )

            response_raw = self._llm.generate_raw_completion(
                user_prompt=prompt,
                system_prompt=system_instruction,
                response_mime_type="application/json"
            )

            payload = json.loads(response_raw.strip())
            raw_steps = payload.get("revised_steps", [])

            revised_steps = []
            for step_data in raw_steps:
                revised_steps.append(PlanStep.model_validate(step_data))

            logger.info("Successfully replanned. Generated %d revised steps.", len(revised_steps))
            return revised_steps

        except Exception as exc:
            logger.error("Replanning failed: %s. Falling back to original remaining steps.", exc)
            # Safe fallback: return original remaining steps to prevent locking up
            return remaining_steps
