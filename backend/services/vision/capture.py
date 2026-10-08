"""
JARVIS Screen Capture Providers.
Implements fast cross-platform screenshot and region capture using 'mss' and Mock fallbacks.
"""

import logging
from PIL import Image

from backend.core.exceptions import ProviderError
from backend.services.vision.interfaces import BaseScreenCaptureProvider

logger = logging.getLogger("jarvis.services.vision.capture")


class MssScreenCaptureProvider(BaseScreenCaptureProvider):
    """Screen capture provider utilizing the mss library for native fast screenshots."""

    def capture_screen(
        self,
        monitor_index: int = 1,
        region: tuple[int, int, int, int] | None = None
    ) -> Image.Image:
        """Captures the full monitor screen or a specified coordinates region.

        Args:
            monitor_index: Index of monitor to capture. Primary is 1.
            region: Optional tuple of (left, top, width, height) pixels.

        Returns:
            PIL Image in RGB format.
        """
        try:
            import mss
        except ImportError as exc:
            raise ProviderError(
                message="The 'mss' package is not installed. Run: pip install mss",
                details={"provider": "mss"}
            ) from exc

        try:
            with mss.mss() as sct:
                if region:
                    # region: (left, top, width, height)
                    left, top, width, height = region
                    monitor = {"left": left, "top": top, "width": width, "height": height}
                else:
                    if monitor_index < 0 or monitor_index >= len(sct.monitors):
                        logger.warning(
                            "Invalid monitor index %d. Falling back to primary (1).",
                            monitor_index
                        )
                        monitor_index = 1
                    monitor = sct.monitors[monitor_index]

                sct_img = sct.grab(monitor)
                # sct_img.bgra contains raw BGRA bytes. Convert to RGB PIL Image.
                return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception as exc:
            logger.error("Failed to capture screen using mss: %s", exc)
            raise ProviderError(
                message=f"Screen capture failed using mss: {exc}",
                details={"monitor_index": monitor_index, "region": region}
            ) from exc


class MockScreenCaptureProvider(BaseScreenCaptureProvider):
    """Mock screen capture provider returning dummy solid-color PIL images for testing."""

    def capture_screen(
        self,
        monitor_index: int = 1,
        region: tuple[int, int, int, int] | None = None
    ) -> Image.Image:
        """Generates a dummy 1920x1080 (or region-sized) slate blue PIL Image."""
        width = region[2] if region else 1920
        height = region[3] if region else 1080
        logger.debug(
            "MockScreenCaptureProvider generating dummy image size %dx%d (region=%s)",
            width, height, region
        )
        return Image.new("RGB", (width, height), color=(30, 41, 59))
