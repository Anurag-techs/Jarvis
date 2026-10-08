"""
JARVIS Desktop Automation Tool.
Structured tool interface to trigger file, terminal, browser, app, and clipboard automation.
Encapsulates explicit user approval.
"""

import logging
from typing import Any

from backend.services.approval import BaseApprovalService
from backend.services.desktop import DesktopService
from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.desktop_automation")


class DesktopAutomationTool(BaseTool):
    """Unified structured tool for executing desktop automation actions under explicit user approval."""

    def __init__(self, desktop_service: DesktopService, approval_service: BaseApprovalService) -> None:
        """Initialize tool.

        Args:
            desktop_service: Low-level OS capabilities service.
            approval_service: User intent approval service.
        """
        self._desktop_service = desktop_service
        self._approval = approval_service

    @property
    def name(self) -> str:
        return "desktop_automation"

    @property
    def description(self) -> str:
        return (
            "Executes OS-level desktop automation commands. "
            "Supported actions: launch_app, create_file, delete_file, list_files, read_file, write_file, "
            "execute_terminal, get_clipboard, set_clipboard, open_browser, search_browser."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "launch_app",
                        "create_file",
                        "delete_file",
                        "list_files",
                        "read_file",
                        "write_file",
                        "execute_terminal",
                        "get_clipboard",
                        "set_clipboard",
                        "open_browser",
                        "search_browser"
                    ],
                    "description": "The target action to execute."
                },
                "args": {
                    "type": "object",
                    "properties": {
                        "app_name": {"type": "string", "description": "Application name to launch"},
                        "file_path": {"type": "string", "description": "Absolute path to file"},
                        "content": {"type": "string", "description": "Text content to write or create"},
                        "directory_path": {"type": "string", "description": "Absolute path to directory to list"},
                        "command": {"type": "string", "description": "Terminal command string to execute"},
                        "text": {"type": "string", "description": "Text value to write to clipboard"},
                        "url": {"type": "string", "description": "Website URL to open in browser"},
                        "query": {"type": "string", "description": "Search query text"}
                    },
                    "description": "Arguments corresponding to the chosen action."
                }
            },
            "required": ["action"]
        }

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        triggers = [
            "launch",
            "open application",
            "run command",
            "execute terminal",
            "clipboard",
            "create file",
            "delete file",
            "read file",
            "write file",
            "list directory"
        ]
        return any(t in lowered for t in triggers)

    def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action")
        args = kwargs.get("args") or {}

        if not action:
            return ToolResult(success=False, message="No action specified for desktop automation.")

        # 1. Determine approval text
        desc = ""
        if action == "launch_app":
            app_name = args.get("app_name") or ""
            desc = f"Launch application '{app_name}'"
        elif action == "create_file":
            file_path = args.get("file_path") or ""
            desc = f"Create new file at '{file_path}'"
        elif action == "delete_file":
            file_path = args.get("file_path") or ""
            desc = f"Permanently delete file/folder at '{file_path}'"
        elif action == "list_files":
            dir_path = args.get("directory_path") or ""
            desc = f"List contents of folder '{dir_path}'"
        elif action == "read_file":
            file_path = args.get("file_path") or ""
            desc = f"Read contents of file '{file_path}'"
        elif action == "write_file":
            file_path = args.get("file_path") or ""
            desc = f"Write/overwrite file at '{file_path}'"
        elif action == "execute_terminal":
            cmd = args.get("command") or ""
            desc = f"Execute terminal command: {cmd}"
        elif action == "get_clipboard":
            desc = "Read current system clipboard contents"
        elif action == "set_clipboard":
            txt = args.get("text") or ""
            snippet = txt if len(txt) <= 40 else txt[:40] + "..."
            desc = f"Set clipboard contents to: '{snippet}'"
        elif action == "open_browser":
            url = args.get("url") or ""
            desc = f"Open URL in web browser: {url}"
        elif action == "search_browser":
            query = args.get("query") or ""
            desc = f"Search Google for: '{query}'"
        else:
            return ToolResult(success=False, message=f"Unsupported action: {action}")

        # 2. Check User Approval
        approved = self._approval.request_approval(desc)
        if not approved:
            logger.warning("Desktop automation action '%s' declined by user.", action)
            return ToolResult(
                success=False,
                message=f"Action declined by user: {desc}",
                error="User declined approval request"
            )

        # 3. Dispatch to service
        success = False
        result_msg = ""
        data = {}

        try:
            if action == "launch_app":
                success, result_msg = self._desktop_service.launch_app(args.get("app_name") or "")
            elif action == "create_file":
                success, result_msg = self._desktop_service.create_file(args.get("file_path") or "", args.get("content") or "")
            elif action == "delete_file":
                success, result_msg = self._desktop_service.delete_file(args.get("file_path") or "")
            elif action == "list_files":
                success, result_msg = self._desktop_service.list_files(args.get("directory_path") or "")
            elif action == "read_file":
                success, result_msg = self._desktop_service.read_file(args.get("file_path") or "")
                if success:
                    data["content"] = result_msg
                    result_msg = f"Successfully read file '{args.get('file_path')}'."
            elif action == "write_file":
                success, result_msg = self._desktop_service.write_file(args.get("file_path") or "", args.get("content") or "")
            elif action == "execute_terminal":
                success, result_msg = self._desktop_service.execute_terminal(args.get("command") or "")
            elif action == "get_clipboard":
                success, result_msg = self._desktop_service.get_clipboard()
                if success:
                    data["text"] = result_msg
                    result_msg = "Successfully read clipboard."
            elif action == "set_clipboard":
                success, result_msg = self._desktop_service.set_clipboard(args.get("text") or "")
            elif action == "open_browser":
                success, result_msg = self._desktop_service.open_browser(args.get("url") or "")
            elif action == "search_browser":
                success, result_msg = self._desktop_service.search_browser(args.get("query") or "")

            return ToolResult(success=success, message=result_msg, data=data)
        except Exception as exc:
            logger.error("Error executing desktop action '%s': %s", action, exc)
            return ToolResult(success=False, message=f"Failed to execute action '{action}': {exc}", error=str(exc))
