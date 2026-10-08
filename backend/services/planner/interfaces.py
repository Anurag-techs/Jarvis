"""
JARVIS Planner Interfaces.
Defines abstract contracts for planners, verifiers, and lifecycle event listeners.
"""

from abc import ABC, abstractmethod
from typing import Any

from backend.services.planner.models import Plan, PlanContext, PlanStep, ExecutionResult


class BasePlanner(ABC):
    """Abstract interface defining the contract for high-level plan generators."""

    @abstractmethod
    def generate_plan(
        self,
        goal: str,
        available_tools: list[dict[str, Any]]
    ) -> list[PlanStep]:
        """Decomposes a high-level goal into a sequence of PlanSteps.

        Args:
            goal: Human-readable target goal.
            available_tools: List of tool schemas registered in the system.

        Returns:
            List of PlanStep models.
        """
        pass


class BaseVerifier(ABC):
    """Abstract interface defining the contract for step outcome verifiers."""

    @abstractmethod
    def verify(self, step: PlanStep, context: PlanContext) -> bool:
        """Verifies if the action outcomes satisfy verification criteria.

        Args:
            step: The executing plan step.
            context: Shared plan variables and execution state.

        Returns:
            True if the verification succeeded, False otherwise.
        """
        pass


class PlanLifecycleListener(ABC):
    """Interface for receiving events throughout the planning & execution lifecycle."""

    def on_plan_start(self, plan: Plan) -> None:
        """Fires when plan execution begins."""
        pass

    def on_plan_complete(self, plan: Plan) -> None:
        """Fires when the plan successfully completes all steps."""
        pass

    def on_plan_failed(self, plan: Plan, error: str) -> None:
        """Fires when the plan halts due to non-recoverable failures."""
        pass

    def on_plan_paused(self, plan: Plan) -> None:
        """Fires when plan execution is paused."""
        pass

    def on_step_start(self, plan: Plan, step: PlanStep) -> None:
        """Fires when execution of a specific step starts."""
        pass

    def on_step_complete(self, plan: Plan, step: PlanStep, result: ExecutionResult) -> None:
        """Fires when a step finishes successfully."""
        pass

    def on_step_failed(self, plan: Plan, step: PlanStep, error: str) -> None:
        """Fires when a step execution fails (even if it will be retried)."""
        pass

    def on_step_skipped(self, plan: Plan, step: PlanStep, reason: str) -> None:
        """Fires when a step is skipped (e.g. unsatisfied dependencies)."""
        pass
