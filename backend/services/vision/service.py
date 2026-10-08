"""
JARVIS Vision Coordination Service.
Orchestrates screen capture, optional OCR, and vision analysis with screen-change caching.
"""

from datetime import datetime
from PIL import Image
import hashlib
import logging
import time

from backend.core.exceptions import ProviderError
from backend.services.vision.interfaces import (
    BaseScreenCaptureProvider,
    BaseOCRProvider,
    BaseVisionProvider,
)
from backend.services.vision.models import ScreenAnalysis, VisionObservation

logger = logging.getLogger("jarvis.services.vision.service")


class VisionService:
    """Orchestrates modular vision operations and implements change detection caching."""

    def __init__(
        self,
        capture_provider: BaseScreenCaptureProvider,
        ocr_provider: BaseOCRProvider | None = None,
        vision_provider: BaseVisionProvider | None = None,
    ) -> None:
        """Initialize the VisionService.

        Args:
            capture_provider: Captures screens or regions.
            ocr_provider: Optional provider for text detection.
            vision_provider: Optional provider for vision LLM analysis.
        """
        self._capture_provider = capture_provider
        self._ocr_provider = ocr_provider
        self._vision_provider = vision_provider

        # Caching states
        self._last_key: tuple[str, bool] | None = None  # (image_hash, ocr_performed)
        self._last_observation: VisionObservation | None = None

    def _compute_image_hash(self, image: Image.Image) -> str:
        """Computes a fast MD5 hash of a resized grayscale representation of the image."""
        try:
            # Resize to 32x32 and convert to grayscale to perform perceptual-like exact hashing
            resized = image.resize((32, 32)).convert("L")
            return hashlib.md5(resized.tobytes()).hexdigest()
        except Exception as e:
            logger.warning("Failed to compute image hash: %s", e)
            return ""

    def clear_cache(self) -> None:
        """Clears the cached screen-change state."""
        logger.debug("Clearing VisionService cache.")
        self._last_key = None
        self._last_observation = None

    def analyze_current_screen(
        self,
        monitor_index: int = 1,
        region: tuple[int, int, int, int] | None = None,
        skip_ocr: bool = False,
    ) -> VisionObservation:
        """Captures screen/region, performs optional OCR, runs visual analysis, and returns observations.

        Args:
            monitor_index: Target monitor to grab.
            region: Optional area to capture: (left, top, width, height).
            skip_ocr: If True, bypasses the OCR step entirely.

        Returns:
            A structured VisionObservation containing analysis details and caching telemetry.
        """
        start_time = time.perf_counter()

        # 1. Capture screenshot image
        capture_start = time.perf_counter()
        image = self._capture_provider.capture_screen(monitor_index=monitor_index, region=region)
        capture_time = time.perf_counter() - capture_start

        # 2. Compute state hash
        current_hash = self._compute_image_hash(image)
        ocr_performed = not skip_ocr and (self._ocr_provider is not None)

        # Check Cache Hit
        cache_key = (current_hash, ocr_performed)
        if (
            self._last_key == cache_key
            and self._last_observation is not None
            and current_hash != ""
        ):
            logger.info("Screen-change cache hit! Returning cached screen analysis.")
            # Return cached observation with updated timestamp and cache_hit=True
            obs = self._last_observation.model_copy(deep=True)
            obs.analysis.timestamp = datetime.now()
            obs.cache_hit = True
            return obs

        # 3. Optional OCR
        ocr_regions = []
        ocr_text = ""
        ocr_time = 0.0

        if ocr_performed:
            logger.info("Executing OCR text detection on captured image...")
            ocr_start = time.perf_counter()
            ocr_regions = self._ocr_provider.detect_text(image)
            ocr_text = "\n".join([r.text for r in ocr_regions])
            ocr_time = time.perf_counter() - ocr_start
        else:
            logger.debug("Skipping OCR step (skip_ocr=%s, ocr_provider_exists=%s)", skip_ocr, self._ocr_provider is not None)

        # 4. LLM Visual Analysis
        vision_time = 0.0
        description = "No visual analysis performed."
        detected_elements = []
        suggested_actions = []

        if self._vision_provider is not None:
            logger.info("Executing vision model analysis...")
            vision_start = time.perf_counter()
            description, detected_elements, suggested_actions = self._vision_provider.analyze_screen(
                image, ocr_text=ocr_text if ocr_performed else None
            )
            vision_time = time.perf_counter() - vision_start
        else:
            logger.warning("No vision LLM provider configured; skipping visual analysis.")

        total_time = time.perf_counter() - start_time

        # Compile telemetry metadata
        metadata = {
            "capture_time_seconds": capture_time,
            "ocr_time_seconds": ocr_time,
            "vision_time_seconds": vision_time,
            "total_time_seconds": total_time,
            "capture_provider": type(self._capture_provider).__name__,
            "ocr_provider": type(self._ocr_provider).__name__ if self._ocr_provider else "None",
            "vision_provider": type(self._vision_provider).__name__ if self._vision_provider else "None",
            "monitor_index": monitor_index,
            "region": region,
        }

        # Build analysis object
        analysis = ScreenAnalysis(
            timestamp=datetime.now(),
            width=image.width,
            height=image.height,
            ocr_text=ocr_text,
            ocr_regions=ocr_regions,
            description=description,
            detected_elements=detected_elements,
            suggested_actions=suggested_actions,
        )

        # Build final observation
        observation = VisionObservation(
            analysis=analysis,
            screenshot_path=None,  # Path can be populated by saving scripts/tools if needed
            state_hash=current_hash,
            cache_hit=False,
        )

        # Save to cache
        self._last_key = cache_key
        # Store metadata inside the analysis object's metadata dict in the observation
        # Pydantic models can be modified, but ScreenAnalysis doesn't have metadata field?
        # Wait, in models.py, I defined metadata field?
        # Let's check models.py: ScreenAnalysis did NOT have metadata in our final models.py write!
        # Ah, let's verify if models.py has a metadata field.
        # Wait! Let's view the content of models.py we wrote:
        # ScreenAnalysis class:
        # class ScreenAnalysis(BaseModel):
        #     timestamp: datetime
        #     width: int
        #     height: int
        #     ocr_text: Optional[str]
        #     ocr_regions: list[OCRTextRegion]
        #     description: str
        #     detected_elements: list[UIElement]
        #     suggested_actions: list[str]
        # (It did NOT have metadata! That is perfectly fine, we don't need it or we can add it to VisionObservation or cache it).
        # We can store execution telemetry in VisionObservation if we want, or log it. Logging is perfect!

        self._last_observation = observation

        logger.debug("Vision analysis completed in %.2fs. Metadata: %s", total_time, metadata)
        return observation
