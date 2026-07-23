"""
JARVIS System Control Tool.

1. Why this module exists:
   Fulfills Version 1.0 requirements for OS control (Shutdown, Restart, Lock PC, Time inquiry).

2. How it fits into the architecture:
   Implements `BaseTool`. Registered in `ToolRegistry`. Delegates to `SystemService` / `backend.utils.system`.

3. Which future modules will interact with it:
   - `backend.tools.registry.ToolRegistry`
   - `backend.services.system_service.SystemService`

4. Common mistakes to avoid:
   - Triggering abrupt destructive OS shutdowns without logging warnings or confirmation flags.

5. Possible future improvements:
   - Timed delayed shutdown (e.g. "Shutdown PC in 30 minutes").
"""

from datetime import datetime
import logging
from typing import Any

from backend.services.system_service import SystemService
from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.system")


class SystemTool(BaseTool):
    """Tool for system OS power and clock commands."""

    def __init__(self, system_service: SystemService | None = None) -> None:
        self._system_service = system_service or SystemService()

    @property
    def name(self) -> str:
        return "system_control"

    @property
    def description(self) -> str:
        return "Controls system OS power states (Shutdown, Restart, Lock PC) and retrieves current local time."

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        triggers = ["shutdown", "restart", "reboot", "lock pc", "lock computer", "what time", "current time"]
        return any(trigger in lowered for trigger in triggers)

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "").lower()
        action = kwargs.get("action")

        if "time" in query or action == "time":
            current_time = datetime.now().strftime("%I:%M %p, %B %d, %Y")
            return ToolResult(
                success=True,
                message=f"The current time is {current_time}.",
                data={"time": current_time},
            )

        if "lock" in query or action == "lock":
            success, msg = self._system_service.lock_pc()
            return ToolResult(success=success, message=msg)

        if "restart" in query or "reboot" in query or action == "restart":
            success, msg = self._system_service.restart_pc()
            return ToolResult(success=success, message=msg)

        if "shutdown" in query or action == "shutdown":
            success, msg = self._system_service.shutdown_pc()
            return ToolResult(success=success, message=msg)

        return ToolResult(
            success=False,
            message="Unrecognized system control action.",
            error=f"Could not map query '{query}' to system action",
        )
