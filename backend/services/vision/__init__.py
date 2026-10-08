"""
JARVIS Vision Package.
Exposes modular components for capturing, extracting text (OCR), and visually analyzing screens.
"""

from backend.services.vision.models import (
    OCRTextRegion,
    UIElement,
    ScreenAnalysis,
    VisionObservation,
)
from backend.services.vision.interfaces import (
    BaseScreenCaptureProvider,
    BaseOCRProvider,
    BaseVisionProvider,
)
from backend.services.vision.capture import (
    MssScreenCaptureProvider,
    MockScreenCaptureProvider,
)
from backend.services.vision.ocr import (
    EasyOCROCRProvider,
    MockOCRProvider,
)
from backend.services.vision.vision_provider import (
    LLMVisionProvider,
    MockVisionProvider,
)
from backend.services.vision.service import VisionService

__all__ = [
    "OCRTextRegion",
    "UIElement",
    "ScreenAnalysis",
    "VisionObservation",
    "BaseScreenCaptureProvider",
    "BaseOCRProvider",
    "BaseVisionProvider",
    "MssScreenCaptureProvider",
    "MockScreenCaptureProvider",
    "EasyOCROCRProvider",
    "MockOCRProvider",
    "LLMVisionProvider",
    "MockVisionProvider",
    "VisionService",
]
