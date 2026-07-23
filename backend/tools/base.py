"""
JARVIS Tool Abstract Base Contract & Result Model.

1. Why this module exists:
   Enforces a strict common interface (`BaseTool`) for all executable capabilities,
   including parameter JSON schemas for AI tool calling.

2. How it fits into the architecture:
   Part of the Tool abstraction layer. SystemOrchestrator and AI Providers interact with tools
   via this contract.

3. Which future modules will interact with it:
   - All concrete tool files (`application_tool`, `weather_tool`, etc.)
   - `backend.tools.registry.ToolRegistry`
   - `backend.tools.executor.ToolExecutor`

4. Common mistakes to avoid:
   - Defining inconsistent tool execution signatures or returning raw unvalidated strings.

5. Possible future improvements:
   - Automated Pydantic model parameter schema extraction.
"""

from abc import ABC, abstractmethod
from typing import Any

from backend.core.models import ToolResult


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

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON schema defining the expected arguments for this tool."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Raw target query or input argument"}
            },
            "required": [],
        }

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Evaluates whether this tool is suitable for handling the given user query."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Executes the tool action using provided parameter arguments."""
