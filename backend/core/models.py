"""
JARVIS Core Domain Data Models.

1. Why this module exists:
   Defines immutable data structures (DTOs) passed between the Orchestrator, AI Providers,
   Tools, and Voice components using Pydantic models.

2. How it fits into the architecture:
   Part of the Core layer. Ensures strict typing, structure validation, and clean schemas
   without direct dependency on external infrastructure.

3. Which future modules will interact with it:
   - backend.core.orchestrator
   - backend.ai.*
   - backend.tools.*
   - backend.services.*
   - backend.api.* (maps these domain models to request/response API schemas)

4. Common mistakes to avoid:
   - Putting business logic or side-effects inside DTO data classes.
   - Using untyped generic dictionaries instead of structured models.

5. Possible future improvements:
   - Extended telemetry metadata fields (latency timing, token usage counters).
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class UserIntent(BaseModel):
    """Structured representation of parsed user intent."""

    raw_text: str = Field(..., description="Original raw transcript or typed input")
    intent_type: Literal["tool_call", "conversation", "unknown"] = Field(
        default="conversation", description="Category of intent detected"
    )
    tool_name: str | None = Field(
        default=None, description="Name of tool to execute if intent_type is tool_call"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to target tool"
    )


class CommandResult(BaseModel):
    """Standardized result wrapper returned by SystemOrchestrator and Tool executions."""

    success: bool = Field(..., description="Whether the operation succeeded")
    response_text: str = Field(..., description="Human-readable text output for TTS/User display")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Structured payload or telemetry data"
    )
    error_message: str | None = Field(default=None, description="Error detail if operation failed")
