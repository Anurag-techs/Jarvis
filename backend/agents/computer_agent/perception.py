"""
Computer Interaction Agent - Perception Component.
Queries VisionService and maps raw UIElements and active window state into WorldState.
"""

import logging
from PIL import Image
from typing import List, Tuple

from backend.agents.computer_agent.interfaces import BasePerception
from backend.agents.computer_agent.models import PerceivedElement, WorldState
from backend.services.vision.service import VisionService

logger = logging.getLogger("jarvis.agents.computer_agent.perception")


class ComputerAgentPerception(BasePerception):
    """Translates VisionService screen analysis and OS focus details into WorldState."""

    def __init__(self, vision_service: VisionService) -> None:
        self._vision_service = vision_service

    def perceive_screen(self, monitor_index: int = 1) -> Tuple[Image.Image, WorldState, str]:
        """Captures screenshot, retrieves active window metadata, and constructs WorldState."""
        logger.info("Executing perception sweep on monitor %d...", monitor_index)

        # Clear Vision Cache to ensure fresh screenshot is processed
        self._vision_service.clear_cache()

        observation = self._vision_service.analyze_current_screen(monitor_index=monitor_index)
        analysis = observation.analysis

        # Capture PIL image via capture provider
        image = self._vision_service._capture_provider.capture_screen(monitor_index=monitor_index)

        # 1. Retrieve Active Window Title (via pygetwindow)
        active_window = "Desktop"
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if win and win.title:
                active_window = win.title
        except Exception as e:
            logger.debug("Failed to retrieve active window title: %s", e)

        # 2. Map perceived elements and compute center coordinates
        perceived: List[PerceivedElement] = []
        for elem in analysis.detected_elements:
            cx, cy = None, None
            if elem.bounding_box and len(elem.bounding_box) == 4:
                x, y, w, h = elem.bounding_box
                cx = x + w // 2
                cy = y + h // 2

            perceived.append(
                PerceivedElement(
                    label=elem.label,
                    semantic_type=elem.semantic_type,
                    bounding_box=elem.bounding_box,
                    confidence=elem.confidence,
                    center_x=cx,
                    center_y=cy,
                )
            )

        # 3. Detect loading indicators/spinners (via OCR keywords and labels)
        has_loading = False
        ocr_lower = analysis.ocr_text.lower() if analysis.ocr_text else ""
        if any(term in ocr_lower for term in ("loading", "wait", "please wait", "spinner", "progress", "processing")):
            has_loading = True
        for elem in perceived:
            lbl = elem.label.lower()
            if any(term in lbl for term in ("loading", "spinner", "progress", "wait")):
                has_loading = True

        world_state = WorldState(
            active_window=active_window,
            visible_elements=perceived,
            ocr_text=analysis.ocr_text or "",
            has_loading_indicator=has_loading,
        )

        logger.info(
            "Perception complete. Active Window: '%s' | UI Elements: %d | Loading Detected: %s",
            world_state.active_window,
            len(world_state.visible_elements),
            world_state.has_loading_indicator,
        )
        return image, world_state, analysis.description


# Backward compatibility alias
VisionPerception = ComputerAgentPerception
