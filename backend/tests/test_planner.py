"""
Unit and integration tests for JARVIS Planning Agent v1.0.
"""

import os
import shutil
import tempfile
import time
import pytest

from backend.tools.registry import ToolRegistry
from backend.tools.executor import ToolExecutor
from backend.ai.provider import MockLLMProvider
from backend.services.planner import (
    Plan,
    PlanStep,
    PlanContext,
    PlanRepository,
    VerificationManager,
    MockVerifier,
    Replanner,
    PlanExecutor,
    MockPlanner,
    PlanningCoordinator,
    PlanLifecycleListener,
)
from backend.tools.planner_tool import PlannerTool


class TrackingListener(PlanLifecycleListener):
    """Tracks fired lifecycle events for testing verification."""

    def __init__(self) -> None:
        self.events = []

    def on_plan_start(self, plan: Plan) -> None:
        self.events.append(("on_plan_start", plan.id))

    def on_plan_complete(self, plan: Plan) -> None:
        self.events.append(("on_plan_complete", plan.id))

    def on_plan_failed(self, plan: Plan, error: str) -> None:
        self.events.append(("on_plan_failed", plan.id, error))

    def on_plan_paused(self, plan: Plan) -> None:
        self.events.append(("on_plan_paused", plan.id))

    def on_step_start(self, plan: Plan, step: PlanStep) -> None:
        self.events.append(("on_step_start", plan.id, step.id))

    def on_step_complete(self, plan: Plan, step: PlanStep, result: Any) -> None:
        self.events.append(("on_step_complete", plan.id, step.id))

    def on_step_failed(self, plan: Plan, step: PlanStep, error: str) -> None:
        self.events.append(("on_step_failed", plan.id, step.id, error))

    def on_step_skipped(self, plan: Plan, step: PlanStep, reason: str) -> None:
        self.events.append(("on_step_skipped", plan.id, step.id, reason))


@pytest.fixture
def temp_plans_dir():
    """Fixture creating a temporary directory for plan persistence."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_plan_context_and_step_dependencies():
    """Verify PlanContext carries state, and PlanStep dependency logic is valid."""
    context = PlanContext()
    context.variables["auth_token"] = "TOKEN_ABC"

    step1 = PlanStep(
        id="step_1",
        description="Fetch Token",
        action_type="wait",
        status="completed"
    )
    step2 = PlanStep(
        id="step_2",
        description="Use Token",
        action_type="tool_call",
        depends_on=["step_1"]
    )

    plan = Plan(id="test_plan", goal="Perform Auth Flow", steps=[step1, step2], context=context)
    assert plan.context.variables["auth_token"] == "TOKEN_ABC"
    assert step2.depends_on == ["step_1"]


def test_plan_persistence(temp_plans_dir):
    """Verify PlanRepository saves and loads plans to JSON correctly."""
    repo = PlanRepository(storage_dir=temp_plans_dir)

    step = PlanStep(id="s1", description="Step 1", action_type="wait")
    plan = Plan(id="p123", goal="Test Goal", steps=[step])

    repo.save_plan(plan)
    assert os.path.exists(os.path.join(temp_plans_dir, "p123.json"))

    loaded = repo.load_plan("p123")
    assert loaded is not None
    assert loaded.id == "p123"
    assert loaded.goal == "Test Goal"
    assert len(loaded.steps) == 1
    assert loaded.steps[0].description == "Step 1"


def test_executor_successful_run():
    """Test full successful plan run with lifecycle event monitoring."""
    registry = ToolRegistry()
    tool_executor = ToolExecutor(registry=registry)
    verifier = VerificationManager([MockVerifier(default_success=True)])
    replanner = Replanner(MockLLMProvider())

    listener = TrackingListener()
    executor = PlanExecutor(
        tool_executor=tool_executor,
        verifier_manager=verifier,
        replanner=replanner,
        listeners=[listener]
    )

    step1 = PlanStep(id="step_1", description="Wait 1", action_type="wait", parameters={"duration": 0})
    step2 = PlanStep(id="step_2", description="Wait 2", action_type="wait", parameters={"duration": 0}, depends_on=["step_1"])
    plan = Plan(id="plan_run", goal="Wait twice", steps=[step1, step2])

    executor.execute(plan, available_tools=[])

    assert plan.status == "completed"
    assert plan.current_step_index == 2
    assert step1.status == "completed"
    assert step2.status == "completed"

    expected_events = [
        ("on_plan_start", "plan_run"),
        ("on_step_start", "plan_run", "step_1"),
        ("on_step_complete", "plan_run", "step_1"),
        ("on_step_start", "plan_run", "step_2"),
        ("on_step_complete", "plan_run", "step_2"),
        ("on_plan_complete", "plan_run")
    ]
    assert listener.events == expected_events


def test_executor_dependencies_skipped():
    """Verify steps are skipped if their depends_on steps are failed/skipped."""
    registry = ToolRegistry()
    tool_executor = ToolExecutor(registry=registry)
    verifier = VerificationManager([MockVerifier(default_success=True)])
    replanner = Replanner(MockLLMProvider())

    listener = TrackingListener()
    executor = PlanExecutor(
        tool_executor=tool_executor,
        verifier_manager=verifier,
        replanner=replanner,
        listeners=[listener]
    )

    # step1 starts failed. step2 depends on step1.
    step1 = PlanStep(id="step_1", description="Step 1", action_type="wait", status="failed")
    step2 = PlanStep(id="step_2", description="Step 2", action_type="wait", depends_on=["step_1"])
    plan = Plan(id="plan_skip", goal="Dependency skip test", steps=[step1, step2])
    plan.current_step_index = 1 # step1 already handled/failed, starting executor on step2

    executor.execute(plan, available_tools=[])

    assert step2.status == "skipped"
    assert plan.status == "completed"  # Plan successfully finishes checking all steps
    assert ("on_step_skipped", "plan_skip", "step_2", "Unsatisfied dependencies") in listener.events


def test_executor_pause_resume():
    """Verify execution can be paused mid-plan and resumed."""
    registry = ToolRegistry()
    tool_executor = ToolExecutor(registry=registry)
    verifier = VerificationManager([MockVerifier(default_success=True)])
    replanner = Replanner(MockLLMProvider())

    listener = TrackingListener()
    executor = PlanExecutor(
        tool_executor=tool_executor,
        verifier_manager=verifier,
        replanner=replanner,
        listeners=[listener]
    )

    step1 = PlanStep(id="step_1", description="Step 1", action_type="wait", parameters={"duration": 0})
    step2 = PlanStep(id="step_2", description="Step 2", action_type="wait", parameters={"duration": 0})
    plan = Plan(id="plan_pause", goal="Pause/Resume Goal", steps=[step1, step2])

    # Pause after step 1 completes using a listener event hook
    class PauseTriggerListener(PlanLifecycleListener):
        def on_step_complete(self, p, s, r):
            if s.id == "step_1":
                executor.pause()

    executor.add_listener(PauseTriggerListener())
    executor.execute(plan, available_tools=[])

    assert plan.status == "paused"
    assert plan.current_step_index == 1
    assert step1.status == "completed"
    assert step2.status == "pending"
    assert ("on_plan_paused", "plan_pause") in listener.events

    # Resume
    executor.resume()
    executor.execute(plan, available_tools=[])

    assert plan.status == "completed"
    assert plan.current_step_index == 2
    assert step2.status == "completed"
    assert ("on_plan_complete", "plan_pause") in listener.events


def test_executor_cancellation():
    """Verify execution halts immediately upon cancel signal."""
    registry = ToolRegistry()
    tool_executor = ToolExecutor(registry=registry)
    verifier = VerificationManager([MockVerifier(default_success=True)])
    replanner = Replanner(MockLLMProvider())

    executor = PlanExecutor(
        tool_executor=tool_executor,
        verifier_manager=verifier,
        replanner=replanner
    )

    step1 = PlanStep(id="step_1", description="Step 1", action_type="wait", parameters={"duration": 0})
    step2 = PlanStep(id="step_2", description="Step 2", action_type="wait", parameters={"duration": 0})
    plan = Plan(id="plan_cancel", goal="Cancel goal", steps=[step1, step2])

    class CancelTriggerListener(PlanLifecycleListener):
        def on_step_complete(self, p, s, r):
            if s.id == "step_1":
                executor.cancel()

    executor.add_listener(CancelTriggerListener())
    executor.execute(plan, available_tools=[])

    assert plan.status == "cancelled"
    assert plan.current_step_index == 1
    assert step2.status == "pending"


def test_executor_retries_and_replanning():
    """Verify step retries occur, and failure triggers replanner steps replacement."""
    from backend.tools.base import BaseTool, ToolResult

    class MockDesktopTool(BaseTool):
        @property
        def name(self) -> str:
            return "desktop_automation"

        @property
        def description(self) -> str:
            return "Mock Tool"

        def can_handle(self, query: str) -> bool:
            return True

        def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, message="Mock success")

    registry = ToolRegistry()
    registry.register(MockDesktopTool())
    tool_executor = ToolExecutor(registry=registry)
    # Verifier always returns False to trigger retry loop and eventual failure
    verifier = VerificationManager([MockVerifier(default_success=False)])
    replanner = Replanner(MockLLMProvider())

    executor = PlanExecutor(
        tool_executor=tool_executor,
        verifier_manager=verifier,
        replanner=replanner
    )

    # Max retries = 1 (meaning 2 attempts total: initial + 1 retry)
    step1 = PlanStep(
        id="step_1",
        description="Fail step",
        action_type="tool_call",
        parameters={"tool_name": "desktop_automation"},
        max_retries=1,
        verification_criteria="Fails always"
    )
    step2 = PlanStep(
        id="step_2",
        description="Next step",
        action_type="tool_call",
        parameters={"tool_name": "desktop_automation"},
        verification_criteria="Fails always"
    )
    plan = Plan(id="plan_replan", goal="Replanning test", steps=[step1, step2])

    executor.execute(plan, available_tools=[])

    # Since step_1 failed verification on all retries, the MockReplanner should have replaced
    # the remaining steps starting from step_2 with f"{step1.id}_alt" alternative step.
    # Therefore, plan.steps became: [step_1, step_1_alt, step_2]
    # And since step_1_alt (the alternative step) will also fail verification (verifier always False),
    # the plan will eventually fail on step_1_alt after retries and no more replan revision.
    assert plan.status == "failed"
    assert len(plan.steps) == 3
    assert plan.steps[1].id == "step_1_alt"
    assert step1.retry_count == 2 # 1 initial + 2 retries (retry incremented to 2, exceeding max_retries 1)


def test_planning_coordinator_and_planner_tool(temp_plans_dir):
    """Verify PlanningCoordinator constructs plans and PlannerTool routes actions correctly."""
    from backend.tools.base import BaseTool, ToolResult

    class MockDesktopAutomationTool(BaseTool):
        @property
        def name(self) -> str:
            return "desktop_automation"

        @property
        def description(self) -> str:
            return "Mock Desktop Automation"

        def can_handle(self, query: str) -> bool:
            return True

        def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, message="Mock desktop action executed successfully")

    registry = ToolRegistry()
    registry.register(MockDesktopAutomationTool())
    tool_executor = ToolExecutor(registry=registry)
    verifier = VerificationManager([MockVerifier(default_success=True)])
    replanner = Replanner(MockLLMProvider())
    executor = PlanExecutor(tool_executor, verifier, replanner)
    repository = PlanRepository(storage_dir=temp_plans_dir)
    planner = MockPlanner()

    coordinator = PlanningCoordinator(planner, executor, repository, registry)
    tool = PlannerTool(coordinator=coordinator)

    # 1. Create plan action
    res_create = tool.execute(action="create_plan", goal="Mock Goal")
    assert res_create.success
    plan_id = res_create.data["id"]
    assert plan_id is not None
    assert len(res_create.data["steps"]) == 2

    # 2. Get plan action
    res_get = tool.execute(action="get_plan", plan_id=plan_id)
    assert res_get.success
    assert res_get.data["status"] == "pending"

    # 3. Execute plan action
    res_exec = tool.execute(action="execute_plan", plan_id=plan_id)
    assert res_exec.success
    assert res_exec.data["status"] == "completed"

    # Verify state was saved to persistence
    loaded = repository.load_plan(plan_id)
    assert loaded is not None
    assert loaded.status == "completed"
