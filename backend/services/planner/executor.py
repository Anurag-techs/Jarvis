"""
JARVIS Plan Executor.
Executes plan steps sequentially with dependency checking, retries, pausing, cancellation, and event hooks.
"""

import logging
import time
from typing import Any, Callable

from backend.core.models import ToolCall
from backend.services.planner.interfaces import PlanLifecycleListener
from backend.services.planner.models import ExecutionResult, Plan, PlanStep
from backend.services.planner.replanner import Replanner
from backend.services.planner.verifier import VerificationManager
from backend.tools.executor import ToolExecutor

logger = logging.getLogger("jarvis.services.planner.executor")


class PlanExecutor:
    """Core engine responsible for running, pausing, and recovering active plans."""

    def __init__(
        self,
        tool_executor: ToolExecutor,
        verifier_manager: VerificationManager,
        replanner: Replanner,
        listeners: list[PlanLifecycleListener] | None = None
    ) -> None:
        """Initialize executor.

        Args:
            tool_executor: Executor driving system tools.
            verifier_manager: Pluggable manager checking step results.
            replanner: Replanning agent for error recovery.
            listeners: List of event listener callbacks.
        """
        self._tool_executor = tool_executor
        self._verifier = verifier_manager
        self._replanner = replanner
        self._listeners = listeners or []
        self._paused = False
        self._cancelled = False

    def add_listener(self, listener: PlanLifecycleListener) -> None:
        """Registers a new lifecycle listener."""
        self._listeners.append(listener)

    def _fire_event(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Dispatches an execution event to all registered listeners."""
        for listener in self._listeners:
            try:
                fn = getattr(listener, event_name, None)
                if fn and callable(fn):
                    fn(*args, **kwargs)
            except Exception as exc:
                logger.error("Error invoking event listener hook %s: %s", event_name, exc)

    def pause(self) -> None:
        """Signals the executor to pause processing before starting the next step."""
        logger.info("Received pause request for execution.")
        self._paused = True

    def resume(self) -> None:
        """Clears the pause signal."""
        logger.info("Received resume request.")
        self._paused = False

    def cancel(self) -> None:
        """Signals the executor to halt processing and mark the plan cancelled."""
        logger.info("Received cancellation request.")
        self._cancelled = True

    def execute(
        self,
        plan: Plan,
        available_tools: list[dict[str, Any]],
        save_callback: Callable[[Plan], None] | None = None
    ) -> None:
        """Executes the plan steps in sequence. Can be resumed if paused.

        Args:
            plan: The Plan instance to execute.
            available_tools: Schemas representing all active tools.
            save_callback: Callback to save plan state on change.
        """
        if plan.status in ("completed", "failed", "cancelled"):
            logger.warning("Plan %s is already in final state: %s", plan.id, plan.status)
            return

        self._paused = False
        self._cancelled = False
        plan.status = "running"
        if save_callback:
            save_callback(plan)

        self._fire_event("on_plan_start", plan)

        while plan.current_step_index < len(plan.steps):
            # 1. Process Pause and Cancel Signals
            if self._cancelled:
                plan.status = "cancelled"
                if save_callback:
                    save_callback(plan)
                self._fire_event("on_plan_failed", plan, "Execution cancelled by user")
                return

            if self._paused:
                plan.status = "paused"
                if save_callback:
                    save_callback(plan)
                self._fire_event("on_plan_paused", plan)
                return

            current_index = plan.current_step_index
            step = plan.steps[current_index]

            # 2. Verify Step Dependencies
            deps_satisfied = True
            for dep_id in step.depends_on:
                dep_step = next((s for s in plan.steps if s.id == dep_id), None)
                if not dep_step or dep_step.status != "completed":
                    deps_satisfied = False
                    break

            if not deps_satisfied:
                logger.warning("Skipping step %s due to unsatisfied dependency list: %s", step.id, step.depends_on)
                step.status = "skipped"
                if save_callback:
                    save_callback(plan)
                self._fire_event("on_step_skipped", plan, step, "Unsatisfied dependencies")
                plan.current_step_index += 1
                continue

            # 3. Begin Step Execution
            step.status = "running"
            if save_callback:
                save_callback(plan)
            self._fire_event("on_step_start", plan, step)

            step_success = False
            step_error = None
            step_result_data = {}

            # Execute step with retries
            while step.retry_count <= step.max_retries and not step_success:
                if self._cancelled:
                    plan.status = "cancelled"
                    if save_callback:
                        save_callback(plan)
                    self._fire_event("on_plan_failed", plan, "Execution cancelled by user")
                    return

                try:
                    logger.info("Executing step %s (attempt %d/%d)", step.id, step.retry_count + 1, step.max_retries + 1)

                    if step.action_type == "tool_call":
                        tool_name = step.parameters.get("tool_name")
                        arguments = step.parameters.get("arguments", {})

                        if not tool_name:
                            raise ValueError(f"Step {step.id} has tool_call action but no tool_name parameter")

                        # Call tool executor
                        tool_call = ToolCall(tool=tool_name, arguments=arguments)
                        tool_results = self._tool_executor.execute_tool_calls([tool_call])
                        tool_res = tool_results[0]

                        if tool_res.success:
                            # Update context variables with output if available before verification
                            plan.context.variables[f"{step.id}_result"] = tool_res.data

                            # Verify step outcome
                            verified = self._verifier.verify(step, plan.context)
                            if verified:
                                step_success = True
                                step_result_data = {"message": tool_res.message, "data": tool_res.data}
                            else:
                                step_error = "Verification criteria not met."
                        else:
                            step_error = tool_res.error or tool_res.message

                    elif step.action_type == "wait":
                        duration = step.parameters.get("duration", 1)
                        logger.info("Waiting %d seconds...", duration)
                        time.sleep(duration)
                        step_success = True
                        step_result_data = {"message": "Wait completed successfully."}

                    else:
                        raise ValueError(f"Unsupported action_type: {step.action_type}")

                except Exception as exc:
                    step_error = str(exc)
                    logger.error("Exception during step %s execution: %s", step.id, exc)

                if not step_success:
                    step.retry_count += 1
                    self._fire_event("on_step_failed", plan, step, step_error or "Unknown failure")
                    if step.retry_count <= step.max_retries:
                        logger.info("Retrying step %s in 1.0 second...", step.id)
                        time.sleep(1.0)

            # 4. Handle Final Step Outcome
            if step_success:
                step.status = "completed"
                step.result = step_result_data
                plan.current_step_index += 1
                if save_callback:
                    save_callback(plan)

                exec_res = ExecutionResult(success=True, message="Completed successfully", data=step_result_data)
                self._fire_event("on_step_complete", plan, step, exec_res)
            else:
                step.status = "failed"
                if save_callback:
                    save_callback(plan)

                # TRIGGER REPLANNING
                if "_alt" in step.id:
                    logger.warning("Alternative step %s failed. Halting plan.", step.id)
                    plan.status = "failed"
                    if save_callback:
                        save_callback(plan)
                    self._fire_event("on_plan_failed", plan, f"Alternative step {step.id} failed: {step_error}")
                    return

                logger.warning("Step %s failed all attempts. Triggering replanner...", step.id)
                try:
                    completed_steps = plan.steps[:current_index]
                    remaining_steps = plan.steps[current_index + 1:]

                    revised_steps = self._replanner.replan(
                        goal=plan.goal,
                        completed_steps=completed_steps,
                        remaining_steps=remaining_steps,
                        failed_step=step,
                        error_message=step_error or "All retries exhausted",
                        available_tools=available_tools
                    )

                    if revised_steps:
                        logger.info("Plan %s steps updated successfully by replanner.", plan.id)
                        plan.steps = completed_steps + [step] + revised_steps
                        plan.current_step_index += 1
                        if save_callback:
                            save_callback(plan)
                        continue
                except Exception as replan_exc:
                    logger.error("Replanning failed: %s", replan_exc)

                plan.status = "failed"
                if save_callback:
                    save_callback(plan)
                self._fire_event("on_plan_failed", plan, f"Step {step.id} failed all retries: {step_error}")
                return

        # 5. Plan Execution Finished
        plan.status = "completed"
        if save_callback:
            save_callback(plan)
        self._fire_event("on_plan_complete", plan)
