"""
JARVIS System & OS Utilities.

1. Why this module exists:
   Provides safe, cross-platform subprocess execution helpers for system commands.

2. How it fits into the architecture:
   Low-level utility module wrapped by system tools and services.

3. Which future modules will interact with it:
   - `backend.tools.system_tool`
   - `backend.tools.application_tool`
   - `backend.tools.browser_tool`
   - `backend.tools.screenshot_tool`

4. Common mistakes to avoid:
   - Executing shell commands without sanitizing arguments (command injection vulnerability).
   - Hardcoding OS-specific commands directly in high-level tool files without OS detection.

5. Possible future improvements:
   - Async subprocess execution with timeout monitors.
"""

import logging
import os
import platform
import subprocess
from typing import Any

logger = logging.getLogger("jarvis.utils.system")


def get_os_type() -> str:
    """Returns normalized OS platform string ('windows', 'darwin', 'linux')."""
    return platform.system().lower()


def run_system_command(command: list[str] | str, shell: bool = False) -> tuple[bool, str]:
    """Executes a system command cleanly and returns success flag and stdout/stderr output.

    Args:
        command: List of command arguments or command string.
        shell: Whether to execute command inside a shell interpreter.

    Returns:
        Tuple of (success_boolean, output_or_error_string).
    """
    logger.debug("Executing OS command: %s (OS: %s)", command, get_os_type())
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        
        logger.warning("OS Command returned non-zero code %d: %s", result.returncode, result.stderr)
        return False, result.stderr.strip() or f"Command exited with status {result.returncode}"
    except Exception as exc:
        logger.error("OS Command execution exception: %s", exc)
        return False, str(exc)
