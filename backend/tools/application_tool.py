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
   - Trusting the LLM to produce Windows-executable app names directly.

5. Possible future improvements:
   - OS application indexer and path resolver.
"""

import logging
import re
from typing import Any

from backend.tools.base import BaseTool, ToolResult
from backend.utils.system import get_os_type, run_system_command

logger = logging.getLogger("jarvis.tools.application")


# ---------------------------------------------------------------------------
# Normalization map: common natural-language names -> Windows executable names
# Keys are lowercase and may contain "open " prefix variants.
# ---------------------------------------------------------------------------
_APP_NAME_MAP: dict[str, str] = {
    "calculator": "calc",
    "calc": "calc",
    "notepad": "notepad",
    "note pad": "notepad",
    "paint": "mspaint",
    "ms paint": "mspaint",
    "mspaint": "mspaint",
    "word": "winword",
    "microsoft word": "winword",
    "excel": "excel",
    "microsoft excel": "excel",
    "powerpoint": "powerpnt",
    "microsoft powerpoint": "powerpnt",
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "mozilla firefox": "firefox",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "explorer": "explorer",
    "file explorer": "explorer",
    "cmd": "cmd",
    "command prompt": "cmd",
    "terminal": "cmd",
    "powershell": "powershell",
    "task manager": "taskmgr",
    "taskmgr": "taskmgr",
    "control panel": "control",
    "settings": "ms-settings:",
    "windows settings": "ms-settings:",
    "snipping tool": "snippingtool",
    "snip": "snippingtool",
    "vlc": "vlc",
    "spotify": "spotify",
    "discord": "discord",
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
}


def _normalize_app_name(raw: str) -> str:
    """Normalizes a natural-language application name to its Windows executable.

    Strips leading action verbs ('open', 'launch', 'start') and performs a
    case-insensitive lookup against the known application map. Falls back to
    the stripped name lowercased if no mapping exists.

    Args:
        raw: Raw application name as received from the LLM (e.g. 'Open Calculator').

    Returns:
        Normalized executable name (e.g. 'calc').
    """
    # Remove leading action verbs that the LLM may inject
    cleaned = re.sub(r"^\s*(open|launch|start|run)\s+", "", raw, flags=re.IGNORECASE).strip()

    lookup_key = cleaned.lower()
    if lookup_key in _APP_NAME_MAP:
        normalized = _APP_NAME_MAP[lookup_key]
        logger.debug("Normalized app name '%s' -> '%s'", raw, normalized)
        return normalized

    # Unknown app: fall back to the stripped name as-is (let OS resolve it)
    logger.debug("No normalization mapping found for '%s'; using stripped name '%s'", raw, cleaned)
    return cleaned


class ApplicationTool(BaseTool):
    """Tool for launching desktop applications with name normalization."""

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
        raw_app_name = kwargs.get("app_name") or kwargs.get("query", "").replace("open", "").replace("launch", "").strip()

        if not raw_app_name:
            return ToolResult(
                success=False,
                message="Please specify which application you would like to open.",
                error="Missing app_name argument",
            )

        # Normalize the natural-language name to a launchable executable
        app_name = _normalize_app_name(raw_app_name)

        os_type = get_os_type()
        logger.info("Attempting to open application '%s' (normalized from '%s') on %s", app_name, raw_app_name, os_type)

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
                message=f"Opening {raw_app_name}.",
                data={"app_name": app_name, "raw_app_name": raw_app_name, "os": os_type},
            )

        return ToolResult(
            success=False,
            message=f"Could not open '{raw_app_name}'. The application may not be installed.",
            error=output,
        )
