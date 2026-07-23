"""
JARVIS Application Launcher Tool.

1. Why this module exists:
   Fulfills Version 1.0 requirement to launch local applications (Notepad, Calculator, Browser, etc.).

2. How it fits into the architecture:
   Implements `BaseTool`. Registered in `ToolRegistry`. Utilizes `backend.utils.system` for execution.

3. Which future modules will interact with it:
   - `backend.tools.registry.ToolRegistry`
   - Future Desktop Automation module in Version 4.0.

4. Common mistakes to avoid:
   - Hardcoding executable file paths for a single OS.

5. Possible future improvements:
   - OS application indexer and path resolver.
"""

import logging
from typing import Any

from backend.tools.base import BaseTool, ToolResult
from backend.utils.system import get_os_type, run_system_command

logger = logging.getLogger("jarvis.tools.application")


class ApplicationTool(BaseTool):
    """Tool for launching desktop applications."""

    @property
    def name(self) -> str:
        return "open_application"

    @property
    def description(self) -> str:
        return "Opens local desktop applications such as Notepad, Calculator, or Terminal."

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        return "open app" in lowered or "launch" in lowered or "open application" in lowered or lowered.startswith("open ")

    def execute(self, **kwargs: Any) -> ToolResult:
        app_name = kwargs.get("app_name") or kwargs.get("query", "").replace("open", "").replace("launch", "").strip()

        if not app_name:
            return ToolResult(
                success=False,
                message="Please specify which application you would like to open.",
                error="Missing app_name argument",
            )

        os_type = get_os_type()
        logger.info("Attempting to open application '%s' on %s", app_name, os_type)

        cmd: list[str]
        if os_type == "windows":
            cmd = ["cmd", "/c", "start", "", app_name]
        elif os_type == "darwin":
            cmd = ["open", "-a", app_name]
        else:
            cmd = [app_name]

        success, output = run_system_command(cmd, shell=(os_type == "windows"))

        if success:
            return ToolResult(
                success=True,
                message=f"Opening application '{app_name}'.",
                data={"app_name": app_name, "os": os_type},
            )
        
        return ToolResult(
            success=False,
            message=f"Could not open application '{app_name}'. Error: {output}",
            error=output,
        )
