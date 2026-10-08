"""
JARVIS Camera Tool.

1. Why this module exists:
   Provides physical webcam access distinct from the screen-capture VisionTool.
   The VisionTool (MssScreenCaptureProvider) only captures the desktop screen.
   This tool captures frames from a physical camera device (webcam) via OpenCV.

2. How it fits into the architecture:
   Implements BaseTool. Registered in ToolRegistry. Uses OpenCV (cv2) for webcam
   capture. Follows the existing multi-action tool pattern (list_cameras,
   capture_camera, analyze_camera).

3. Which future modules will interact with it:
   - backend.tools.registry.ToolRegistry
   - backend.core.startup.StartupManager (registration)
   - IntentRouter (routing camera queries)

4. Common mistakes to avoid:
   - Confusing this webcam tool with VisionTool (which is screen capture only).
   - Leaving camera handles open after an error (always release in finally).
   - Assuming camera index 0 is always available without enumeration.

5. Possible future improvements:
   - Configurable camera index via settings.
   - Stream mode for continuous capture.
   - Integration with LLMVisionProvider for semantic analysis of webcam frames.
"""

import logging
import os
from datetime import datetime
from typing import Any

from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.camera")


def _enumerate_cameras(max_index: int = 5) -> list:
    """Safely enumerates available camera devices via OpenCV.

    Args:
        max_index: Maximum camera index to probe (0-based).

    Returns:
        List of dicts: [{index, resolution}, ...]
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("OpenCV (cv2) is not installed. Cannot enumerate cameras.")
        return []

    cameras = []
    for idx in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({"index": idx, "resolution": f"{width}x{height}"})
        except Exception as probe_err:
            logger.debug("Camera probe error at index %d: %s", idx, probe_err)
        finally:
            if cap is not None:
                cap.release()

    return cameras


class CameraTool(BaseTool):
    """Multi-action tool for physical webcam access.

    Distinct from VisionTool which only captures the desktop screen.
    This tool accesses physical camera hardware.

    Actions:
    - list_cameras: Enumerates available camera devices.
    - capture_camera: Captures a single frame and saves it to disk.
    - analyze_camera: Captures a frame and describes it via LLM vision.
    """

    def __init__(self, camera_index: int = 0, save_dir: str = "screenshots") -> None:
        self._default_camera_index = camera_index
        self._save_dir = save_dir

    @property
    def name(self) -> str:
        return "camera_actions"

    @property
    def description(self) -> str:
        return (
            "Controls the physical webcam/camera device. "
            "Actions: 'list_cameras' (enumerate available cameras), "
            "'capture_camera' (capture a single frame and save to file), "
            "'analyze_camera' (capture frame and describe what is visible)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_cameras", "capture_camera", "analyze_camera"],
                    "description": (
                        "The camera action to perform: "
                        "'list_cameras' (find available devices), "
                        "'capture_camera' (save a webcam frame), "
                        "'analyze_camera' (capture + describe the scene)."
                    ),
                },
                "camera_index": {
                    "type": "integer",
                    "description": "Camera device index to use. Defaults to 0.",
                    "default": 0,
                },
            },
            "required": ["action"],
        }

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        triggers = [
            "camera", "webcam", "take a photo", "take photo",
            "capture photo", "what does the camera see", "look through camera",
        ]
        return any(t in lowered for t in triggers)

    def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "capture_camera")
        camera_index = int(kwargs.get("camera_index", self._default_camera_index))

        if action == "list_cameras":
            return self._action_list_cameras()
        elif action == "capture_camera":
            return self._action_capture(camera_index=camera_index)
        elif action == "analyze_camera":
            return self._action_analyze(camera_index=camera_index)
        else:
            return ToolResult(
                success=False,
                message=(
                    f"Unsupported camera action: '{action}'. "
                    "Valid actions: list_cameras, capture_camera, analyze_camera."
                ),
            )

    # ------------------------------------------------------------------
    # Private action implementations
    # ------------------------------------------------------------------

    def _action_list_cameras(self) -> ToolResult:
        cameras = _enumerate_cameras()
        if not cameras:
            return ToolResult(
                success=False,
                message=(
                    "No compatible camera device was detected. "
                    "Check that a webcam is connected and that no other application "
                    "is currently using it."
                ),
                data={"cameras": []},
            )
        camera_list = ", ".join(
            f"Camera {c['index']} ({c['resolution']})" for c in cameras
        )
        return ToolResult(
            success=True,
            message=f"Found {len(cameras)} camera(s): {camera_list}.",
            data={"cameras": cameras},
        )

    def _action_capture(self, camera_index: int = 0) -> ToolResult:
        """Captures a single frame from the webcam and saves it."""
        try:
            import cv2  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                message="OpenCV is not installed. Run: pip install opencv-python",
                error="ImportError: cv2",
            )

        cap = None
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(camera_index)  # fallback without backend

            if not cap.isOpened():
                return ToolResult(
                    success=False,
                    message=(
                        f"No compatible camera device was detected at index {camera_index}. "
                        "Check that a webcam is connected and Windows Camera privacy is "
                        "enabled (Settings > Privacy > Camera)."
                    ),
                    error=f"VideoCapture({camera_index}) failed to open",
                )

            ret, frame = cap.read()
            if not ret or frame is None:
                return ToolResult(
                    success=False,
                    message="Camera opened but failed to capture a frame. The device may be in use.",
                    error="cap.read() returned False",
                )

            os.makedirs(self._save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self._save_dir, f"camera_{timestamp}.png")
            cv2.imwrite(filepath, frame)

            h, w = frame.shape[:2]
            logger.info("Camera frame captured: %s (%dx%d)", filepath, w, h)

            return ToolResult(
                success=True,
                message=f"Camera frame captured and saved to {filepath} ({w}x{h}).",
                data={"filepath": filepath, "camera_index": camera_index, "width": w, "height": h},
            )

        except Exception as exc:
            logger.error("CameraTool capture failed: %s", exc)
            return ToolResult(
                success=False,
                message=f"Camera capture failed: {exc}",
                error=str(exc),
            )
        finally:
            if cap is not None:
                cap.release()

    def _action_analyze(self, camera_index: int = 0) -> ToolResult:
        """Captures a frame and analyzes it using LLM vision (gracefully degrades)."""
        capture_result = self._action_capture(camera_index=camera_index)
        if not capture_result.success:
            return capture_result

        filepath = capture_result.data.get("filepath", "")

        try:
            from PIL import Image  # type: ignore
            import google.generativeai as genai  # type: ignore

            img = Image.open(filepath)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                ["Describe what you see in this webcam image in 2-3 sentences.", img]
            )
            description = response.text.strip() if response.text else "No description returned."
            logger.info("Camera analyze description: %.120s", description)

            return ToolResult(
                success=True,
                message=f"Camera analysis: {description}",
                data={"filepath": filepath, "description": description, "camera_index": camera_index},
            )
        except Exception as exc:
            logger.warning("CameraTool analyze: LLM vision unavailable (%s). Returning capture only.", exc)
            return ToolResult(
                success=True,
                message=(
                    f"Camera frame captured and saved to {filepath}. "
                    "LLM vision analysis is currently unavailable."
                ),
                data={"filepath": filepath, "camera_index": camera_index, "description": None},
            )
