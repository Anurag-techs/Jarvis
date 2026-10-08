"""
JARVIS Planner Data Models.
Defines Pydantic models for structured planning agent context, steps, and execution states.
"""

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """Represents a single step in an execution plan."""

    id: str = Field(..., description="Unique step identifier (e.g. 'step_1')")
    description: str = Field(..., description="Human-readable description of what this step does")
    action_type: str = Field(..., description="Action category (e.g., 'tool_call', 'wait')")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments required for this step's execution (e.g., tool_name, arguments)"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of steps that must successfully complete before this step is executed"
    )
    status: Literal["pending", "running", "completed", "failed", "skipped"] = Field(
        default="pending",
        description="Operational status of the step"
    )
    verification_criteria: Optional[str] = Field(
        default=None,
        description="Criteria statement evaluated via VerificationManager to confirm outcomes"
    )
    result: Optional[dict[str, Any]] = Field(
        default=None,
        description="Execution results or telemetry collected during execution"
    )
    retry_count: int = Field(default=0, description="Current retry attempts count")
    max_retries: int = Field(default=3, description="Maximum retries permitted")


class PlanContext(BaseModel):
    """Holds variables, outcomes, and states shared across all step executions in a plan."""

    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value variables passed between planning steps"
    )
    system_state: dict[str, Any] = Field(
        default_factory=dict,
        description="System state metrics collected during execution"
    )
    start_time: datetime = Field(default_factory=datetime.now)


class Plan(BaseModel):
    """Structured representation of a high-level goal and its execution steps."""

    id: str = Field(..., description="Unique plan identifier")
    goal: str = Field(..., description="The overall goal being accomplished")
    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered list of plan steps"
    )
    status: Literal["pending", "running", "paused", "completed", "failed", "cancelled"] = Field(
        default="pending",
        description="Operational status of the overall plan"
    )
    current_step_index: int = Field(
        default=0,
        description="Index of the currently active/executing step"
    )
    context: PlanContext = Field(
        default_factory=PlanContext,
        description="Shared state context across step execution"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ExecutionResult(BaseModel):
    """Result return wrapper for a plan step execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    message: str = Field(..., description="Short status description")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Outcome parameters or collected metrics"
    )
    error: Optional[str] = Field(default=None, description="Detail error string if failed")
