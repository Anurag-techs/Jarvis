"""
JARVIS Planner Tool.
Structured tool interface allowing the system orchestrator to create, execute, pause, resume, cancel, or query plans.
"""

import logging
from typing import Any

from backend.services.planner.coordinator import PlanningCoordinator
from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.planner")


class PlannerTool(BaseTool):
    """Unified tool interface exposing JARVIS Planning Agent capabilities."""

    def __init__(self, coordinator: PlanningCoordinator) -> None:
        """Initialize the PlannerTool.

        Args:
            coordinator: Central planning coordinator.
        """
        self._coordinator = coordinator

    @property
    def name(self) -> str:
        return "planner_actions"

    @property
    def description(self) -> str:
        return (
            "Performs goal planning and step execution coordination. "
            "Actions: 'create_plan' (goal -> steps), 'execute_plan' (run steps), "
            "'pause_plan' (halt execution), 'resume_plan' (continue execution), "
            "'cancel_plan' (permanently abort plan), 'get_plan' (fetch status)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create_plan",
                        "execute_plan",
                        "pause_plan",
                        "resume_plan",
                        "cancel_plan",
                        "get_plan"
                    ],
                    "description": "The planning operational action to run."
                },
                "goal": {
                    "type": "string",
                    "description": "The target high-level goal statement. Required for 'create_plan'."
                },
                "plan_id": {
                    "type": "string",
                    "description": "Target plan identifier. Required for execute, pause, resume, cancel, or get actions."
                }
            },
            "required": ["action"]
        }

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        triggers = [
            "create plan",
            "execute plan",
            "pause plan",
            "resume plan",
            "cancel plan",
            "run goal",
            "planner status"
        ]
        return any(t in lowered for t in triggers)

    def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action")
        goal = kwargs.get("goal")
        plan_id = kwargs.get("plan_id")

        if not action:
            return ToolResult(success=False, message="No action specified for PlannerTool.")

        try:
            if action == "create_plan":
                if not goal:
                    return ToolResult(success=False, message="Parameter 'goal' is required for create_plan.")
                plan = self._coordinator.create_plan(goal)
                return ToolResult(
                    success=True,
                    message=f"Plan {plan.id} created successfully with {len(plan.steps)} steps.",
                    data=plan.model_dump()
                )

            elif action == "execute_plan":
                if not plan_id:
                    return ToolResult(success=False, message="Parameter 'plan_id' is required for execute_plan.")
                self._coordinator.execute_plan(plan_id)
                plan = self._coordinator.get_plan(plan_id)
                status = plan.status if plan else "unknown"
                return ToolResult(
                    success=True,
                    message=f"Execution of plan {plan_id} terminated with status: {status}",
                    data=plan.model_dump() if plan else {}
                )

            elif action == "pause_plan":
                if not plan_id:
                    return ToolResult(success=False, message="Parameter 'plan_id' is required for pause_plan.")
                self._coordinator.pause_plan(plan_id)
                return ToolResult(
                    success=True,
                    message=f"Sent pause signal for plan {plan_id}."
                )

            elif action == "resume_plan":
                if not plan_id:
                    return ToolResult(success=False, message="Parameter 'plan_id' is required for resume_plan.")
                self._coordinator.resume_plan(plan_id)
                plan = self._coordinator.get_plan(plan_id)
                status = plan.status if plan else "unknown"
                return ToolResult(
                    success=True,
                    message=f"Resumed execution of plan {plan_id}. Execution status: {status}",
                    data=plan.model_dump() if plan else {}
                )

            elif action == "cancel_plan":
                if not plan_id:
                    return ToolResult(success=False, message="Parameter 'plan_id' is required for cancel_plan.")
                self._coordinator.cancel_plan(plan_id)
                return ToolResult(
                    success=True,
                    message=f"Sent cancellation signal for plan {plan_id}."
                )

            elif action == "get_plan":
                if not plan_id:
                    return ToolResult(success=False, message="Parameter 'plan_id' is required for get_plan.")
                plan = self._coordinator.get_plan(plan_id)
                if not plan:
                    return ToolResult(success=False, message=f"Plan with ID {plan_id} not found.")
                return ToolResult(
                    success=True,
                    message=f"Retrieved plan {plan_id} with status: {plan.status}",
                    data=plan.model_dump()
                )

            else:
                return ToolResult(success=False, message=f"Unsupported action: {action}")

        except Exception as exc:
            logger.error("PlannerTool action %s failed: %s", action, exc)
            return ToolResult(success=False, message=f"Planner action '{action}' failed: {exc}", error=str(exc))
