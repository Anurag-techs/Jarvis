"""
JARVIS Tool Abstract Base Contract & Result Model.

1. Why this module exists:
   Enforces a strict common interface (`BaseTool`) for all executable capabilities.
   Guarantees that every tool provides name, description, trigger capabilities, and execution signature.

2. How it fits into the architecture:
   Part of the Tool abstraction layer. SystemOrchestrator interacts with tools exclusively
   via this common interface.

3. Which future modules will interact with it:
   - All concrete tool files (`application_tool`, `weather_tool`, etc.)
   - `backend.tools.registry.ToolRegistry`
   - Future Version 6 plugin architecture.

4. Common mistakes to avoid:
   - Defining inconsistent tool execution signatures or returning raw unvalidated strings.

5. Possible future improvements:
   - JSON Schema auto-generation from tool parameter annotations for LLM function calling.
"""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standard output schema returned by every tool execution."""

    success: bool = Field(..., description="Whether tool execution succeeded")
    message: str = Field(..., description="Human readable result summary for user feedback")
    data: dict[str, Any] = Field(default_factory=dict, description="Structured return payload")
    error: str | None = Field(default=None, description="Error detail if execution failed")


class BaseTool(ABC):
    """Abstract Base Class for all JARVIS tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name of the tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable explanation of what the tool accomplishes."""

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Evaluates whether this tool is suitable for handling the given user query."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Executes the tool action using provided parameter arguments."""
