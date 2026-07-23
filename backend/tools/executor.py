"""
JARVIS Tool Executor Service.

1. Why this module exists:
   Isolates tool execution logic from the core orchestrator. Responsible for validating tools,
   checking arguments, executing tool calls, capturing errors, and preparing the pipeline for future
   execution retries, approvals, and parallel async execution.

2. How it fits into the architecture:
   Part of `backend.tools`. Injected into `SystemOrchestrator`.

3. Which future modules will interact with it:
   - `backend.core.orchestrator.SystemOrchestrator`
   - Future Tool Approval & Security Manager.

4. Common mistakes to avoid:
   - Mixing tool execution loops directly inside orchestrator or UI code.

5. Possible future improvements:
   - Parallel asyncio execution (`asyncio.gather`) for multi-tool requests.
   - User approval prompts for high-risk system commands.
"""

import logging

from backend.core.models import ToolCall, ToolResult
from backend.tools.registry import ToolRegistry

logger = logging.getLogger("jarvis.tools.executor")


class ToolExecutor:
    """Service executing and managing validation for ToolCall instances."""

    def __init__(self, registry: ToolRegistry) -> None:
        """Initialize ToolExecutor via dependency injection.

        Args:
            registry: Injected ToolRegistry instance.
        """
        self._registry = registry

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Validates and executes a single ToolCall instance.

        Args:
            tool_call: ToolCall instance containing tool name and arguments.

        Returns:
            ToolResult containing execution success state and return payload.
        """
        logger.info("ToolExecutor validating tool call: %s (args: %s)", tool_call.tool, tool_call.arguments)
        
        tool = self._registry.get_tool(tool_call.tool)
        if not tool:
            logger.warning("ToolExecutor: Tool '%s' requested by AI is not registered", tool_call.tool)
            return ToolResult(
                success=False,
                message=f"Tool '{tool_call.tool}' is not available on this system.",
                error=f"Tool {tool_call.tool} not registered",
            )

        # TODO: Add interactive user approval check for dangerous actions (e.g. Shutdown / System Restart)
        # TODO: Add retry handler with exponential backoff for transient tool execution failures

        return self._registry.execute_tool(tool_call.tool, tool_call.arguments)

    def execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Executes a list of ToolCall instances sequentially.

        Args:
            tool_calls: List of ToolCall objects.

        Returns:
            List of ToolResult output schemas.
        """
        results: list[ToolResult] = []
        for call in tool_calls:
            result = self.execute_tool_call(call)
            results.append(result)
        return results
