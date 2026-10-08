"""
JARVIS OCR Providers.
Implements swappable text extraction using EasyOCR and Mock fallbacks.
"""

import logging
from PIL import Image

from backend.core.exceptions import ProviderError
from backend.services.vision.interfaces import BaseOCRProvider
from backend.services.vision.models import OCRTextRegion

logger = logging.getLogger("jarvis.services.vision.ocr")


class EasyOCROCRProvider(BaseOCRProvider):
    """OCR Provider using the EasyOCR deep-learning library."""

    def __init__(self, languages: list[str] | None = None) -> None:
        """Initialize the EasyOCR provider.

        Args:
            languages: List of language codes (e.g. ['en']). Defaults to ['en'].
        """
        self._languages = languages or ["en"]
        self._reader = None

    def _get_reader(self):
        """Lazy-initializes and caches the EasyOCR Reader instance."""
        if self._reader is None:
            try:
                import easyocr
                # gpu=True will auto-fallback to CPU if CUDA is not available
                logger.info("Initializing EasyOCR Reader for languages: %s", self._languages)
                self._reader = easyocr.Reader(self._languages, gpu=True)
            except ImportError as exc:
                raise ProviderError(
                    message=(
                        "EasyOCR or PyTorch is not installed. "
                        "Run: pip install easyocr torch torchvision"
                    ),
                    details={"provider": "easyocr"}
                ) from exc
            except Exception as exc:
                logger.error("Failed to initialize EasyOCR Reader: %s", exc)
                raise ProviderError(
                    message=f"Failed to initialize EasyOCR Reader: {exc}",
                    details={"languages": self._languages}
                ) from exc
        return self._reader

    def detect_text(self, image: Image.Image) -> list[OCRTextRegion]:
        """Runs EasyOCR text detection on the PIL Image.

        Args:
            image: PIL Image to analyze.

        Returns:
            List of OCRTextRegion models.
        """
        reader = self._get_reader()

        try:
            import numpy as np

            # Convert PIL Image to numpy array (RGB) for EasyOCR
            img_np = np.array(image)

            # readtext returns: [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], text, confidence), ...]
            raw_results = reader.readtext(img_np)

            regions = []
            for bbox, text, confidence in raw_results:
                # Convert coordinate structures to native Python ints
                bbox_py = [[int(pt[0]), int(pt[1])] for pt in bbox]
                regions.append(
                    OCRTextRegion(
                        text=text,
                        confidence=float(confidence),
                        bounding_box=bbox_py
                    )
                )

            logger.debug("EasyOCR detected %d text regions.", len(regions))
            return regions
        except Exception as exc:
            logger.error("EasyOCR text detection failed: %s", exc)
            raise ProviderError(
                message=f"OCR detection failed: {exc}",
                details={"provider": "easyocr"}
            ) from exc


class MockOCRProvider(BaseOCRProvider):
    """Mock OCR provider returning predefined text blocks for offline testing."""

    def detect_text(self, image: Image.Image) -> list[OCRTextRegion]:
        """Returns standard mockup text regions representing a template screen."""
        logger.debug("MockOCRProvider returning simulated screen text regions.")
        return [
            OCRTextRegion(
                text="Welcome to JARVIS",
                confidence=0.98,
                bounding_box=[[50, 50], [250, 50], [250, 80], [50, 80]]
            ),
            OCRTextRegion(
                text="Status: Online",
                confidence=0.95,
                bounding_box=[[50, 100], [180, 100], [180, 120], [50, 120]]
            ),
            OCRTextRegion(
                text="Click here to login",
                confidence=0.92,
                bounding_box=[[400, 300], [550, 300], [550, 330], [400, 330]]
            )
        ]
