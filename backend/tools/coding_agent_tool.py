"""
JARVIS Coding Agent Tool.

1. Why this module exists:
   Wraps the CodingAgent coordinator as a BaseTool so it can be registered in the
   ToolRegistry and invoked via the standard tool execution pipeline.

2. How it fits into the architecture:
   Registered in StartupManager.  NOT responsible for its own intent detection —
   routing is handled exclusively by IntentRouter inside SystemOrchestrator.
   This tool does NOT implement can_handle(); the base class abstract method is
   satisfied with a stub that always returns False to make explicit that routing
   is external.

3. Which future modules will interact with it:
   - backend.tools.registry.ToolRegistry
   - backend.tools.executor.ToolExecutor

4. Common mistakes to avoid:
   - Adding intent detection logic (keyword matching) inside this tool.
   - Importing VisionService or DesktopService directly; inject them via constructor.

5. Possible future improvements:
   - Expose preferred_language and mode as tool parameters so the LLM can drive them.
"""

import logging
from typing import Any

from backend.agents.coding_agent.coordinator import CodingAgent
from backend.agents.coding_agent.models import CodingGoal, CodingMode, ProgrammingLanguage
from backend.core.models import ToolResult
from backend.tools.base import BaseTool

logger = logging.getLogger("jarvis.tools.coding_agent")


class CodingAgentTool(BaseTool):
    """Tool wrapper for the CodingAgent pipeline.

    Receives high-level coding goals and delegates them to the CodingAgent,
    which handles screen capture, parsing, generation, validation, insertion,
    and verification internally.
    """

    def __init__(self, coding_agent: CodingAgent) -> None:
        """Initialize with a pre-built CodingAgent (dependency injection).

        Args:
            coding_agent: Fully wired CodingAgent coordinator instance.
        """
        self._agent = coding_agent

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "coding_actions"

    @property
    def description(self) -> str:
        return (
            "Solve coding problems visible on screen, explain MCQs, "
            "generate code, and insert the solution into the active editor. "
            "Use when the user asks to solve a LeetCode/HackerRank problem, "
            "answer a multiple-choice question, or write code for an on-screen task."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["solve", "mcq"],
                    "description": "Operating mode: 'solve' for coding problems, 'mcq' for multiple-choice questions.",
                },
                "language": {
                    "type": "string",
                    "description": "Preferred programming language (e.g. 'python', 'java', 'cpp'). Defaults to 'python'.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional high-level description of the goal (used for logging).",
                },
            },
            "required": [],
        }

    def can_handle(self, query: str) -> bool:
        """Intentionally not used — routing is done by IntentRouter, not the tool."""
        return False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the CodingAgent pipeline.

        Args:
            mode: 'solve' | 'mcq'  (default: 'solve')
            language: preferred programming language  (default: 'python')
            description: optional goal description string

        Returns:
            ToolResult with success flag, human-readable message, and rich data payload.
        """
        # Parse mode
        raw_mode = str(kwargs.get("mode", "solve")).lower().strip()
        try:
            mode = CodingMode(raw_mode)
        except ValueError:
            mode = CodingMode.SOLVE

        # Parse language
        raw_lang = str(kwargs.get("language", "python")).lower().strip()
        try:
            language = ProgrammingLanguage(raw_lang)
        except ValueError:
            language = ProgrammingLanguage.PYTHON

        description = str(kwargs.get("description", "Solve the visible problem"))

        goal = CodingGoal(
            description=description,
            mode=mode,
            preferred_language=language,
        )

        logger.info(
            "CodingAgentTool.execute() — mode=%s | language=%s | description=%r",
            mode.value, language.value, description[:80],
        )

        result = self._agent.execute(goal)

        return ToolResult(
            success=result.success,
            message=result.message,
            data={
                "mode": result.mode.value,
                "steps_taken": result.steps_taken,
                "duration_seconds": result.duration_seconds,
                "problem_title": result.problem.title if result.problem else None,
                "solution_language": result.solution.language.value if result.solution else None,
                "time_complexity": result.solution.time_complexity if result.solution else None,
                "space_complexity": result.solution.space_complexity if result.solution else None,
                "explanation": result.solution.explanation if result.solution else None,
                "validation_passed": result.validation.is_valid if result.validation else None,
                "mcq_correct_answer": (
                    f"{result.mcq_explanation.correct_label}: {result.mcq_explanation.correct_text}"
                    if result.mcq_explanation else None
                ),
            },
            error=None if result.success else result.message,
        )
