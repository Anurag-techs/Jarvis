"""
JARVIS Tool Registry.

1. Why this module exists:
   Implements the Registry Pattern to manage pluggable tools and expose tool metadata schemas for AI Tool Calling.

2. How it fits into the architecture:
   Part of the Tool infrastructure layer. Injected into `SystemOrchestrator` and `ToolExecutor`.

3. Which future modules will interact with it:
   - `backend.tools.executor.ToolExecutor`
   - `backend.conversation.system_prompt`

4. Common mistakes to avoid:
   - Hardcoding tool instantiation inside the core orchestrator loop instead of using `register()`.

5. Possible future improvements:
   - Dynamic hot-reloading of tools from a designated plugins directory.
"""

import logging
from typing import Any

from backend.core.exceptions import ToolExecutionError
from backend.core.models import ToolResult
from backend.tools.base import BaseTool

logger = logging.getLogger("jarvis.tools.registry")


class ToolRegistry:
    """Central registry storing and managing available BaseTool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance.

        Args:
            tool: Instance subclassing BaseTool.
        """
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool registration for: %s", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s (%s)", tool.name, tool.description)

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """Retrieves registered tool by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        """Returns list of registered tool names."""
        return list(self._tools.keys())

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Returns list of registered tool metadata dictionaries including name, description, and parameters schema."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            }
            for tool in self._tools.values()
        ]

    def __len__(self) -> int:
        """Returns number of registered tools."""
        return len(self._tools)

    def execute_tool(self, tool_name: str, parameters: dict[str, Any]) -> ToolResult:
        """Looks up and executes target tool by name.

        Args:
            tool_name: Name identifier of tool.
            parameters: Arguments dictionary.

        Returns:
            ToolResult payload.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error("Tool execution failed: Tool '%s' not registered", tool_name)
            return ToolResult(
                success=False,
                message=f"Tool '{tool_name}' is not available.",
                error=f"Tool {tool_name} not found in registry",
            )

        try:
            logger.info("Executing tool '%s' with params: %s", tool_name, parameters)
            return tool.execute(**parameters)
        except Exception as exc:
            logger.error("Unhandled exception during execution of tool '%s': %s", tool_name, exc)
            raise ToolExecutionError(
                message=f"Tool '{tool_name}' failed to execute: {exc}",
                details={"tool_name": tool_name, "parameters": parameters},
            ) from exc
