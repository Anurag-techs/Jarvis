"""
Integration tests for JARVIS Desktop Automation v1.0.
Verifies file operations, app launching, clipboard, browser action, command execution, and approval logic.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from backend.core.models import ToolCall, ToolResult
from backend.services.approval import AutoApprovalService
from backend.services.desktop import DesktopService
from backend.tools.desktop_automation_tool import DesktopAutomationTool


class TestDesktopAutomation(unittest.TestCase):
    def setUp(self) -> None:
        self.desktop_service = DesktopService()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_user_approval_declined(self) -> None:
        """Verifies that if user approval is declined, no action runs and tool returns failure."""
        approval = AutoApprovalService(approved=False)
        tool = DesktopAutomationTool(self.desktop_service, approval)

        result = tool.execute(
            action="create_file",
            args={
                "file_path": os.path.join(self.temp_dir, "test.txt"),
                "content": "Hello World"
            }
        )

        self.assertFalse(result.success)
        self.assertIn("Action declined by user", result.message)
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "test.txt")))

    @patch("subprocess.Popen")
    def test_launch_app_success(self, mock_popen: MagicMock) -> None:
        """Verifies launch_app resolves normalizations and spawns process with approval."""
        approval = AutoApprovalService(approved=True)
        tool = DesktopAutomationTool(self.desktop_service, approval)

        result = tool.execute(
            action="launch_app",
            args={"app_name": "Calculator"}
        )

        self.assertTrue(result.success)
        self.assertIn("launched successfully", result.message)
        mock_popen.assert_called_once()

    def test_file_operations_lifecycle(self) -> None:
        """Verifies create, list, read, write, and delete operations on local workspace files."""
        approval = AutoApprovalService(approved=True)
        tool = DesktopAutomationTool(self.desktop_service, approval)

        target_file = os.path.join(self.temp_dir, "test_file.txt")

        # 1. Create file
        res = tool.execute(
            action="create_file",
            args={"file_path": target_file, "content": "Initial content"}
        )
        self.assertTrue(res.success)
        self.assertTrue(os.path.exists(target_file))

        # 2. Prevent duplicate creation
        res_dup = tool.execute(
            action="create_file",
            args={"file_path": target_file, "content": "Duplicate"}
        )
        self.assertFalse(res_dup.success)
        self.assertIn("already exists", res_dup.message)

        # 3. Read file
        res_read = tool.execute(
            action="read_file",
            args={"file_path": target_file}
        )
        self.assertTrue(res_read.success)
        self.assertEqual(res_read.data.get("content"), "Initial content")

        # 4. Write/Overwrite file
        res_write = tool.execute(
            action="write_file",
            args={"file_path": target_file, "content": "Updated content"}
        )
        self.assertTrue(res_write.success)

        # Verify updated read
        res_read2 = tool.execute(
            action="read_file",
            args={"file_path": target_file}
        )
        self.assertEqual(res_read2.data.get("content"), "Updated content")

        # 5. List directory contents
        res_list = tool.execute(
            action="list_files",
            args={"directory_path": self.temp_dir}
        )
        self.assertTrue(res_list.success)
        self.assertIn("test_file.txt", res_list.message)

        # 6. Delete file
        res_del = tool.execute(
            action="delete_file",
            args={"file_path": target_file}
        )
        self.assertTrue(res_del.success)
        self.assertFalse(os.path.exists(target_file))

    def test_execute_terminal_command(self) -> None:
        """Verifies executing simple terminal command returns output."""
        approval = AutoApprovalService(approved=True)
        tool = DesktopAutomationTool(self.desktop_service, approval)

        # Execute simple platform independent echoing
        res = tool.execute(
            action="execute_terminal",
            args={"command": "echo Hello_JARVIS"}
        )
        self.assertTrue(res.success)
        self.assertIn("Hello_JARVIS", res.message)

    @patch("webbrowser.open")
    def test_browser_actions(self, mock_web_open: MagicMock) -> None:
        """Verifies opening URLs and searches triggers standard browser actions."""
        approval = AutoApprovalService(approved=True)
        tool = DesktopAutomationTool(self.desktop_service, approval)

        # test open URL
        res = tool.execute(
            action="open_browser",
            args={"url": "google.com"}
        )
        self.assertTrue(res.success)
        mock_web_open.assert_any_call("https://google.com")

        # test browser search
        res_search = tool.execute(
            action="search_browser",
            args={"query": "test query"}
        )
        self.assertTrue(res_search.success)
        mock_web_open.assert_any_call("https://www.google.com/search?q=test query")

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_clipboard_operations(self, mock_popen: MagicMock, mock_run: MagicMock) -> None:
        """Verifies set_clipboard and get_clipboard are invoked correctly using OS specific commands."""
        approval = AutoApprovalService(approved=True)
        tool = DesktopAutomationTool(self.desktop_service, approval)

        # Mock sub-process return values
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        mock_proc.communicate.return_value = ("", "")

        mock_completed = MagicMock()
        mock_completed.stdout = "Copied text"
        mock_completed.returncode = 0
        mock_run.return_value = mock_completed

        # Test set clipboard
        res_set = tool.execute(
            action="set_clipboard",
            args={"text": "Test Clip"}
        )
        self.assertTrue(res_set.success)

        # Test get clipboard
        res_get = tool.execute(
            action="get_clipboard"
        )
        self.assertTrue(res_get.success)
        self.assertEqual(res_get.data.get("text"), "Copied text")


if __name__ == "__main__":
    unittest.main()
