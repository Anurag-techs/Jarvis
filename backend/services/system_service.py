"""
JARVIS System Service.

1. Why this module exists:
   Encapsulates OS-level system administration capabilities (Shutdown, Restart, Lock PC).

2. How it fits into the architecture:
   Part of the Service layer. Interacts directly with cross-platform OS subprocess utilities.

3. Which future modules will interact with it:
   - `backend.tools.system_tool.SystemTool`
   - `backend.core.orchestrator.SystemOrchestrator`

4. Common mistakes to avoid:
   - Hardcoding single-OS binary execution strings without inspecting platform.system().

5. Possible future improvements:
   - System resource utilization diagnostics (CPU %, RAM, Disk usage).
"""

import logging

from backend.utils.system import get_os_type, run_system_command

logger = logging.getLogger("jarvis.services.system")


class SystemService:
    """Service executing operating system lifecycle actions."""

    def lock_pc(self) -> tuple[bool, str]:
        """Locks the workstation across Windows, macOS, or Linux."""
        os_type = get_os_type()
        logger.info("Executing PC lock command for OS: %s", os_type)

        cmd: list[str]
        if os_type == "windows":
            cmd = ["rundll32.exe", "user32.dll,LockWorkStation"]
        elif os_type == "darwin":
            cmd = ["pmset", "displaysleepnow"]
        else:
            cmd = ["loginctl", "lock-session"]

        success, output = run_system_command(cmd)
        if success:
            return True, "Locking PC workstation."
        return False, f"Failed to lock PC: {output}"

    def shutdown_pc(self, delay_seconds: int = 0) -> tuple[bool, str]:
        """Initiates system shutdown."""
        os_type = get_os_type()
        logger.warning("Initiating system shutdown for OS: %s", os_type)

        cmd: list[str]
        if os_type == "windows":
            cmd = ["shutdown", "/s", "/t", str(delay_seconds)]
        elif os_type == "darwin":
            cmd = ["sudo", "shutdown", "-h", "now"]
        else:
            cmd = ["shutdown", "-h", "now"]

        success, output = run_system_command(cmd)
        if success:
            return True, f"Initiating system shutdown in {delay_seconds} seconds."
        return False, f"Failed to initiate shutdown: {output}"

    def restart_pc(self, delay_seconds: int = 0) -> tuple[bool, str]:
        """Initiates system restart/reboot."""
        os_type = get_os_type()
        logger.warning("Initiating system reboot for OS: %s", os_type)

        cmd: list[str]
        if os_type == "windows":
            cmd = ["shutdown", "/r", "/t", str(delay_seconds)]
        else:
            cmd = ["shutdown", "-r", "now"]

        success, output = run_system_command(cmd)
        if success:
            return True, f"Initiating system restart in {delay_seconds} seconds."
        return False, f"Failed to initiate restart: {output}"
