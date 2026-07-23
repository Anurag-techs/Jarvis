"""
JARVIS Core Domain Data Models.

1. Why this module exists:
   Defines immutable data structures (DTOs) passed between the Orchestrator, AI Providers,
   Tools, Conversation Manager, and User Interfaces using Pydantic models.

2. How it fits into the architecture:
   Part of the Core layer. Ensures strict typing, structure validation, and clean schemas
   without direct dependency on external infrastructure.

3. Which future modules will interact with it:
   - backend.core.orchestrator
   - backend.tools.executor
   - backend.conversation.manager
   - backend.ai.*

4. Common mistakes to avoid:
   - Putting business logic or side-effects inside DTO data classes.
   - Using untyped generic dictionaries instead of structured models.

5. Possible future improvements:
   - Extended telemetry metadata fields (latency timing, token usage counters).
"""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """Structured model representing a single turn in a conversation."""

    role: Literal["user", "assistant", "system"] = Field(..., description="Role sender of message")
    content: str = Field(..., description="Text content of message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of creation")


class ToolCall(BaseModel):
    """Structured model representing an AI decision to invoke a specific tool."""

    tool: str = Field(..., description="Target tool name identifier")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments dictionary for execution")


class ToolResult(BaseModel):
    """Standardized output schema returned by every tool execution."""

    success: bool = Field(..., description="Whether tool execution succeeded")
    message: str = Field(..., description="Human readable summary or result string")
    data: dict[str, Any] = Field(default_factory=dict, description="Structured return payload")
    error: str | None = Field(default=None, description="Error detail if execution failed")


class AssistantResponse(BaseModel):
    """Structured response model returned by AI Providers and SystemOrchestrator."""

    text: str = Field(..., description="Human-readable text output to display to the user")
    should_speak: bool = Field(default=True, description="Whether this response should be spoken via TTS")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Requested tool execution calls")
    success: bool = Field(default=True, description="Whether processing succeeded")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Payload metadata or telemetry")
    error: str | None = Field(default=None, description="Error details if command processing failed")


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
    """Legacy result wrapper returned by Tool executions."""

    success: bool = Field(..., description="Whether the operation succeeded")
    response_text: str = Field(..., description="Human-readable text output for TTS/User display")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Structured payload or telemetry data"
    )
    error_message: str | None = Field(default=None, description="Error detail if operation failed")
