"""
JARVIS Vision Module Interfaces.
Defines abstract base classes for swappable vision providers.
"""

from abc import ABC, abstractmethod
from PIL import Image

from backend.services.vision.models import OCRTextRegion, UIElement


class BaseScreenCaptureProvider(ABC):
    """Abstract interface defining contract for capturing screens or regions."""

    @abstractmethod
    def capture_screen(
        self,
        monitor_index: int = 1,
        region: tuple[int, int, int, int] | None = None
    ) -> Image.Image:
        """Captures either the whole screen or a sub-region.

        Args:
            monitor_index: Target monitor to capture (default: 1).
            region: Optional tuple of (left, top, width, height) to capture a specific area.

        Returns:
            PIL Image of the captured screen region.
        """
        pass


class BaseOCRProvider(ABC):
    """Abstract interface defining contract for OCR text extraction."""

    @abstractmethod
    def detect_text(self, image: Image.Image) -> list[OCRTextRegion]:
        """Detects text regions in the provided PIL Image.

        Args:
            image: PIL Image object.

        Returns:
            List of OCRTextRegion objects.
        """
        pass


class BaseVisionProvider(ABC):
    """Abstract interface defining contract for LLM visual screen analysis."""

    @abstractmethod
    def analyze_screen(
        self,
        image: Image.Image,
        ocr_text: str | None = None
    ) -> tuple[str, list[UIElement], list[str]]:
        """Analyzes a screen image using a vision LLM model.

        Args:
            image: PIL Image of the screen.
            ocr_text: Optional helper OCR-detected text.

        Returns:
            Tuple containing:
                - description: A text description of the screen.
                - detected_elements: A list of UIElement models.
                - suggested_actions: A list of suggested next actions.
        """
        pass
