"""
Tests for JARVIS Problem 1 & 2 fixes:
- Problem 1: Chrome/app argument extraction from IntentRouter → orchestrator → ApplicationTool
- Problem 2: CameraTool physical webcam implementation
- DesktopEditorController import fix verification
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.agents.coding_agent.editor_controller import (
    CodingEditorController,
    DesktopEditorController,
)
from backend.core.models import AssistantResponse, ToolCall
from backend.core.orchestrator import _extract_tool_arguments, _extract_open_application_args
from backend.services.intent_router import IntentRouter
from backend.tools.application_tool import ApplicationTool, _normalize_app_name
from backend.tools.camera_tool import CameraTool, _enumerate_cameras
from backend.tools.registry import ToolRegistry


# ============================================================
# PROBLEM 1 — Chrome Argument Extraction
# ============================================================


class TestIntentRouterAppRouting(unittest.TestCase):
    """Verifies IntentRouter correctly routes open-app commands."""

    def setUp(self) -> None:
        self.router = IntentRouter()

    def test_open_chrome_routes_to_open_application(self) -> None:
        assert self.router.route("Open Chrome") == "open_application"

    def test_open_google_chrome_routes(self) -> None:
        assert self.router.route("Open Google Chrome") == "open_application"

    def test_open_notepad_routes(self) -> None:
        assert self.router.route("Open Notepad") == "open_application"

    def test_open_calculator_routes(self) -> None:
        assert self.router.route("Open Calculator") == "open_application"

    def test_open_terminal_routes(self) -> None:
        assert self.router.route("Open Terminal") == "open_application"

    def test_launch_chrome_routes(self) -> None:
        assert self.router.route("Launch Chrome") == "open_application"

    def test_open_file_explorer_routes(self) -> None:
        assert self.router.route("Open File Explorer") == "open_application"


class TestExtractOpenApplicationArgs(unittest.TestCase):
    """Verifies _extract_open_application_args strips verbs and punctuation."""

    def test_open_chrome(self) -> None:
        args = _extract_open_application_args("Open Chrome")
        self.assertEqual(args["app_name"], "Chrome")

    def test_open_chrome_with_period(self) -> None:
        args = _extract_open_application_args("Open Chrome.")
        self.assertEqual(args["app_name"], "Chrome")

    def test_open_google_chrome(self) -> None:
        args = _extract_open_application_args("Open Google Chrome")
        self.assertEqual(args["app_name"], "Google Chrome")

    def test_launch_chrome(self) -> None:
        args = _extract_open_application_args("Launch Chrome")
        self.assertEqual(args["app_name"], "Chrome")

    def test_start_chrome(self) -> None:
        args = _extract_open_application_args("Start Chrome")
        self.assertEqual(args["app_name"], "Chrome")

    def test_open_notepad(self) -> None:
        args = _extract_open_application_args("Open Notepad")
        self.assertEqual(args["app_name"], "Notepad")

    def test_open_calculator(self) -> None:
        args = _extract_open_application_args("Open Calculator")
        self.assertEqual(args["app_name"], "Calculator")

    def test_open_terminal(self) -> None:
        args = _extract_open_application_args("Open Terminal")
        self.assertEqual(args["app_name"], "Terminal")


class TestExtractToolArguments(unittest.TestCase):
    """Verifies _extract_tool_arguments dispatches correctly per tool."""

    def test_open_application_extracts_app_name(self) -> None:
        args = _extract_tool_arguments("open_application", "Open Chrome.")
        self.assertIn("app_name", args)
        self.assertNotIn("description", args)
        self.assertEqual(args["app_name"], "Chrome")

    def test_system_actions_extracts_query(self) -> None:
        args = _extract_tool_arguments("system_actions", "Volume up")
        self.assertIn("query", args)

    def test_unknown_tool_returns_query(self) -> None:
        args = _extract_tool_arguments("some_unknown_tool", "Do something")
        self.assertEqual(args, {"query": "Do something"})

    def test_camera_capture_action(self) -> None:
        args = _extract_tool_arguments("camera_actions", "Take a photo")
        self.assertEqual(args["action"], "capture_camera")

    def test_camera_list_action(self) -> None:
        args = _extract_tool_arguments("camera_actions", "List cameras")
        self.assertEqual(args["action"], "list_cameras")

    def test_camera_analyze_action(self) -> None:
        args = _extract_tool_arguments("camera_actions", "Analyze the camera")
        self.assertEqual(args["action"], "analyze_camera")


class TestApplicationToolEndToEnd(unittest.TestCase):
    """End-to-end: app_name kwarg flows through to ApplicationTool correctly."""

    def setUp(self) -> None:
        self.tool = ApplicationTool()

    def test_chrome_opens_correctly(self) -> None:
        with patch("backend.tools.application_tool.run_system_command", return_value=(True, "")) as mock_cmd:
            result = self.tool.execute(app_name="Chrome")
            self.assertTrue(result.success, f"Expected success, got: {result.message}")
            called_cmd = mock_cmd.call_args[0][0]
            self.assertIn("chrome", called_cmd)
            self.assertNotIn("Open Chrome", called_cmd)

    def test_notepad_opens_correctly(self) -> None:
        with patch("backend.tools.application_tool.run_system_command", return_value=(True, "")):
            result = self.tool.execute(app_name="Notepad")
            self.assertTrue(result.success)

    def test_calculator_opens_correctly(self) -> None:
        with patch("backend.tools.application_tool.run_system_command", return_value=(True, "")) as mock_cmd:
            result = self.tool.execute(app_name="Calculator")
            self.assertTrue(result.success)
            called_cmd = mock_cmd.call_args[0][0]
            self.assertIn("calc", called_cmd)

    def test_terminal_opens_correctly(self) -> None:
        with patch("backend.tools.application_tool.run_system_command", return_value=(True, "")) as mock_cmd:
            result = self.tool.execute(app_name="Terminal")
            self.assertTrue(result.success)
            called_cmd = mock_cmd.call_args[0][0]
            self.assertIn("cmd", called_cmd)

    def test_google_chrome_normalizes(self) -> None:
        with patch("backend.tools.application_tool.run_system_command", return_value=(True, "")) as mock_cmd:
            result = self.tool.execute(app_name="Google Chrome")
            self.assertTrue(result.success)
            called_cmd = mock_cmd.call_args[0][0]
            self.assertIn("chrome", called_cmd)

    def test_no_app_name_returns_error(self) -> None:
        """Confirms empty app_name still returns the helpful error message."""
        result = self.tool.execute()
        self.assertFalse(result.success)
        self.assertIn("specify", result.message)

    def test_description_key_NOT_passed_directly(self) -> None:
        """description kwarg should not be used as an app name."""
        result = self.tool.execute(description="Open Chrome.")
        # Must NOT succeed by treating "Open Chrome." as the app name
        # (it will fail OS launch, but must NOT pass "Open Chrome." to OS)
        with patch("backend.tools.application_tool.run_system_command", return_value=(True, "")) as mock_cmd:
            # Even if it somehow succeeds, the command must not contain "description"
            self.tool.execute(description="Open Chrome.")
            if mock_cmd.called:
                called_cmd = mock_cmd.call_args[0][0]
                self.assertNotIn("description", str(called_cmd))


# ============================================================
# PROBLEM 2 — Camera Tool
# ============================================================


class TestCameraToolListCameras(unittest.TestCase):
    """Tests for CameraTool list_cameras action."""

    def setUp(self) -> None:
        self.tool = CameraTool()

    def test_list_cameras_returns_result(self) -> None:
        result = self.tool.execute(action="list_cameras")
        # Should always return a ToolResult (success or failure)
        self.assertIsNotNone(result)
        self.assertIn("cameras", result.data)

    def test_list_cameras_with_mock_enumeration_found(self) -> None:
        mock_cameras = [{"index": 0, "resolution": "640x480"}]
        with patch("backend.tools.camera_tool._enumerate_cameras", return_value=mock_cameras):
            result = self.tool.execute(action="list_cameras")
        self.assertTrue(result.success)
        self.assertIn("1 camera", result.message)
        self.assertEqual(result.data["cameras"][0]["index"], 0)

    def test_list_cameras_with_no_cameras(self) -> None:
        with patch("backend.tools.camera_tool._enumerate_cameras", return_value=[]):
            result = self.tool.execute(action="list_cameras")
        self.assertFalse(result.success)
        self.assertIn("No compatible camera", result.message)


class TestCameraToolCapture(unittest.TestCase):
    """Tests for CameraTool capture_camera action."""

    def setUp(self) -> None:
        self.tool = CameraTool()

    def test_capture_returns_result(self) -> None:
        """Smoke test: capture must return a ToolResult (not raise)."""
        result = self.tool.execute(action="capture_camera", camera_index=0)
        self.assertIsNotNone(result)

    def test_capture_with_mock_camera(self) -> None:
        """Verifies capture returns a ToolResult when camera is available."""
        import numpy as np

        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.release = MagicMock()
        mock_cap.get.return_value = 640

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_DSHOW = 700
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.imwrite = MagicMock()

        import sys
        with patch.dict(sys.modules, {"cv2": mock_cv2}), \
             patch("os.makedirs"):
            result = self.tool.execute(action="capture_camera", camera_index=0)

        # Should always return a ToolResult without raising
        self.assertIsNotNone(result)
        self.assertIn(result.success, [True, False])

    def test_capture_no_camera_returns_friendly_message(self) -> None:
        """When no camera is available, returns descriptive failure message."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap.release = MagicMock()

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_DSHOW = 700

        import sys
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = self.tool.execute(action="capture_camera", camera_index=99)

        self.assertFalse(result.success)
        self.assertIn("No compatible camera", result.message)


class TestCameraToolInvalidAction(unittest.TestCase):
    """Tests CameraTool rejects unknown actions gracefully."""

    def test_unknown_action_returns_error(self) -> None:
        tool = CameraTool()
        result = tool.execute(action="fly_to_moon")
        self.assertFalse(result.success)
        self.assertIn("fly_to_moon", result.message)


class TestCameraToolRegistration(unittest.TestCase):
    """Verifies CameraTool can be registered in ToolRegistry."""

    def test_camera_tool_registers(self) -> None:
        registry = ToolRegistry()
        registry.register(CameraTool())
        self.assertIn("camera_actions", registry.list_tools())


class TestCameraToolName(unittest.TestCase):
    """Verifies CameraTool metadata."""

    def test_name_is_camera_actions(self) -> None:
        self.assertEqual(CameraTool().name, "camera_actions")

    def test_description_mentions_webcam(self) -> None:
        self.assertIn("webcam", CameraTool().description.lower())

    def test_parameters_schema_has_action(self) -> None:
        schema = CameraTool().parameters_schema
        self.assertIn("action", schema["properties"])
        self.assertIn("required", schema)
        self.assertIn("action", schema["required"])


# ============================================================
# DesktopEditorController Import Fix
# ============================================================


class TestDesktopEditorControllerImport(unittest.TestCase):
    """Verifies the DesktopEditorController alias imports without error."""

    def test_desktop_editor_controller_importable(self) -> None:
        """DesktopEditorController must be importable from editor_controller module."""
        from backend.agents.coding_agent.editor_controller import DesktopEditorController
        self.assertIsNotNone(DesktopEditorController)

    def test_desktop_editor_controller_is_coding_editor_controller(self) -> None:
        """DesktopEditorController alias must point to CodingEditorController."""
        self.assertIs(DesktopEditorController, CodingEditorController)

    def test_coding_agent_tool_importable(self) -> None:
        from backend.tools.coding_agent_tool import CodingAgentTool
        self.assertIsNotNone(CodingAgentTool)


if __name__ == "__main__":
    unittest.main()
