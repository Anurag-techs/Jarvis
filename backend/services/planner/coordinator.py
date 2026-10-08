"""
JARVIS Planning Coordinator.
Act as the orchestrating control plane above high-level planners, executors, and persistence layers.
"""

import logging
import uuid
from typing import Optional

from backend.services.planner.interfaces import BasePlanner, PlanLifecycleListener
from backend.services.planner.models import Plan
from backend.services.planner.persistence import PlanRepository
from backend.services.planner.executor import PlanExecutor
from backend.tools.registry import ToolRegistry

logger = logging.getLogger("jarvis.services.planner.coordinator")


class PlanningCoordinator:
    """Central manager facilitating plan lifecycle operations and DI wiring."""

    def __init__(
        self,
        planner: BasePlanner,
        executor: PlanExecutor,
        repository: PlanRepository,
        tool_registry: ToolRegistry,
    ) -> None:
        """Initialize the PlanningCoordinator.

        Args:
            planner: Pluggable high-level plan generator.
            executor: Step-by-step executor loop engine.
            repository: Local plan state filesystem repository.
            tool_registry: Registry to inspect available capabilities.
        """
        self._planner = planner
        self._executor = executor
        self._repository = repository
        self._tools = tool_registry

    def add_listener(self, listener: PlanLifecycleListener) -> None:
        """Registers a listener to monitor step and plan lifecycle state changes."""
        self._executor.add_listener(listener)

    def create_plan(self, goal: str) -> Plan:
        """Decomposes goal, constructs a Plan state, and persists it.

        Args:
            goal: User target goal.

        Returns:
            The created Plan instance.
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        logger.info("Creating new plan %s for goal: '%s'", plan_id, goal)

        available_tool_schemas = self._tools.get_available_tools()
        steps = self._planner.generate_plan(goal, available_tool_schemas)

        plan = Plan(id=plan_id, goal=goal, steps=steps)
        self._repository.save_plan(plan)
        return plan

    def execute_plan(self, plan_id: str) -> None:
        """Loads and drives execution of a plan to completion (or pause/cancellation).

        Args:
            plan_id: Target plan.
        """
        plan = self._repository.load_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan with ID {plan_id} not found")

        logger.info("Starting execution of plan %s", plan_id)
        available_tool_schemas = self._tools.get_available_tools()

        # Execute using shared repository callback to save state on every transition
        self._executor.execute(
            plan=plan,
            available_tools=available_tool_schemas,
            save_callback=self._repository.save_plan
        )

    def pause_plan(self, plan_id: str) -> None:
        """Signals pause to execution loop."""
        logger.info("Pausing plan %s", plan_id)
        self._executor.pause()

    def resume_plan(self, plan_id: str) -> None:
        """Signals resume and runs plan execution."""
        logger.info("Resuming plan %s", plan_id)
        self._executor.resume()
        self.execute_plan(plan_id)

    def cancel_plan(self, plan_id: str) -> None:
        """Signals cancellation to active execution loop."""
        logger.info("Cancelling plan %s", plan_id)
        self._executor.cancel()

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Fetches plan state.

        Args:
            plan_id: Plan ID.

        Returns:
            Plan state or None.
        """
        return self._repository.load_plan(plan_id)

    def list_plans(self) -> list[Plan]:
        """Lists all stored plans."""
        return self._repository.list_plans()
