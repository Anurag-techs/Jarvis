"""
JARVIS Vision Tool.
Structured multi-action tool interface to capture, run OCR, or visually analyze screens.
"""

import logging
from typing import Any

from backend.services.vision.service import VisionService
from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.vision")


class VisionTool(BaseTool):
    """Multi-action tool for native screen capture, OCR, and semantic visual layout analysis."""

    def __init__(self, vision_service: VisionService) -> None:
        """Initialize the VisionTool with the shared VisionService.

        Args:
            vision_service: Coordinate service for capture, OCR, and Vision APIs.
        """
        self._vision_service = vision_service

    @property
    def name(self) -> str:
        return "vision_actions"

    @property
    def description(self) -> str:
        return (
            "Performs screen visual actions. "
            "Actions: 'capture' (take and save screenshot), "
            "'ocr' (extract text from screen/region), "
            "'analyze' (generate structured semantic layout analysis of screen/region)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["capture", "ocr", "analyze"],
                    "description": (
                        "The vision action to execute: "
                        "'capture' (saves screenshot file), "
                        "'ocr' (extracts text), "
                        "'analyze' (runs LLM vision model layout description)."
                    )
                },
                "monitor_index": {
                    "type": "integer",
                    "description": "Index of monitor to target. Primary monitor is 1.",
                    "default": 1
                },
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Optional bounding region coordinate values: [left, top, width, height] in pixels."
                },
                "skip_ocr": {
                    "type": "boolean",
                    "description": "For 'analyze' action: Bypasses the OCR extraction step if set to True.",
                    "default": False
                }
            },
            "required": ["action"]
        }

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        triggers = [
            "visual analysis",
            "analyze screen",
            "what's on screen",
            "describe screen",
            "read text from screen",
            "ocr screen"
        ]
        return any(t in lowered for t in triggers)

    def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action")
        monitor_index = kwargs.get("monitor_index", 1)
        region_list = kwargs.get("region")
        skip_ocr = kwargs.get("skip_ocr", False)

        region = tuple(region_list) if region_list else None

        if not action:
            return ToolResult(success=False, message="No action specified for VisionTool.")

        if action == "capture":
            try:
                import os
                from datetime import datetime

                os.makedirs("screenshots", exist_ok=True)
                filename = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

                image = self._vision_service._capture_provider.capture_screen(
                    monitor_index=monitor_index,
                    region=region
                )
                image.save(filename)
                logger.info("Saved capture screenshot to %s", filename)

                return ToolResult(
                    success=True,
                    message=f"Screen captured and saved to {filename} ({image.width}x{image.height}).",
                    data={"filepath": filename, "width": image.width, "height": image.height}
                )
            except Exception as e:
                logger.error("VisionTool capture action failed: %s", e)
                return ToolResult(success=False, message=f"Capture failed: {e}", error=str(e))

        elif action == "ocr":
            try:
                if not self._vision_service._ocr_provider:
                    return ToolResult(
                        success=False,
                        message="OCR Provider is not configured in VisionService."
                    )

                image = self._vision_service._capture_provider.capture_screen(
                    monitor_index=monitor_index,
                    region=region
                )
                regions = self._vision_service._ocr_provider.detect_text(image)
                ocr_text = "\n".join([r.text for r in regions])
                regions_data = [r.model_dump() for r in regions]

                return ToolResult(
                    success=True,
                    message=f"OCR completed. Detected {len(regions)} text blocks.",
                    data={"ocr_text": ocr_text, "regions": regions_data}
                )
            except Exception as e:
                logger.error("VisionTool OCR action failed: %s", e)
                return ToolResult(success=False, message=f"OCR failed: {e}", error=str(e))

        elif action == "analyze":
            try:
                observation = self._vision_service.analyze_current_screen(
                    monitor_index=monitor_index,
                    region=region,
                    skip_ocr=skip_ocr
                )
                analysis_data = observation.model_dump()
                msg = (
                    f"Visual analysis complete. Cache hit: {observation.cache_hit}.\n"
                    f"Description: {observation.analysis.description}\n"
                    f"UI Elements detected: {len(observation.analysis.detected_elements)}\n"
                    f"Suggested actions: {len(observation.analysis.suggested_actions)}"
                )
                return ToolResult(
                    success=True,
                    message=msg,
                    data=analysis_data
                )
            except Exception as e:
                logger.error("VisionTool visual analysis failed: %s", e)
                return ToolResult(success=False, message=f"Visual analysis failed: {e}", error=str(e))

        else:
            return ToolResult(success=False, message=f"Unsupported action: {action}")
