"""
JARVIS Screenshot Tool.

1. Why this module exists:
   Fulfills Version 1.0 requirement to capture a screenshot of the user's primary desktop display.

2. How it fits into the architecture:
   Implements `BaseTool`. Registered in `ToolRegistry`. Uses cross-platform standard library / subprocess fallback.

3. Which future modules will interact with it:
   - `backend.tools.registry.ToolRegistry`
   - Future Vision Module (Version 3.0) for visual desktop analysis.

4. Common mistakes to avoid:
   - Attempting to load heavy third-party vision libraries in V1.0 foundation.

5. Possible future improvements:
   - Specific window selection and multi-monitor capture handles.
"""

from datetime import datetime
import logging
import os
from typing import Any

from backend.tools.base import BaseTool, ToolResult
from backend.utils.system import get_os_type, run_system_command

logger = logging.getLogger("jarvis.tools.screenshot")


class ScreenshotTool(BaseTool):
    """Tool for taking desktop screenshots."""

    @property
    def name(self) -> str:
        return "take_screenshot"

    @property
    def description(self) -> str:
        return "Captures a screenshot of the current screen display."

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        return "screenshot" in lowered or "capture screen" in lowered or "take screenshot" in lowered

    def execute(self, **kwargs: Any) -> ToolResult:
        os_type = get_os_type()
        os.makedirs("screenshots", exist_ok=True)
        filename = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        logger.info("Capturing screenshot to %s on platform %s", filename, os_type)

        success = False
        output = ""

        if os_type == "windows":
            # PowerShell command using System.Drawing to capture primary screen
            ps_cmd = (
                "[Reflection.Assembly]::LoadWithPartialName('System.Drawing'); "
                "$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
                "$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; "
                "$graphics = [System.Drawing.Graphics]::FromImage($bmp); "
                "$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); "
                f"$bmp.Save('{filename}'); "
                "$graphics.Dispose(); $bmp.Dispose();"
            )
            success, output = run_system_command(["powershell", "-Command", ps_cmd])
        elif os_type == "darwin":
            success, output = run_system_command(["screencapture", filename])
        else:
            success, output = run_system_command(["import", "-window", "root", filename])

        if success or os.path.exists(filename):
            return ToolResult(
                success=True,
                message=f"Screenshot successfully saved to {filename}.",
                data={"filepath": filename},
            )

        return ToolResult(
            success=False,
            message="Failed to take screenshot.",
            error=output,
        )
