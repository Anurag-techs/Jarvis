"""
JARVIS Vision Module Data Models.
Defines Pydantic models for structured visual state analysis and planner compatibility.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class OCRTextRegion(BaseModel):
    """Represents a text region detected by OCR."""

    text: str = Field(..., description="The text content detected")
    confidence: float = Field(..., description="The confidence score of the detection")
    bounding_box: list[list[int]] = Field(
        ...,
        description="Bounding box coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]"
    )


class UIElement(BaseModel):
    """Represents a UI element detected by the vision model on the screen."""

    label: str = Field(..., description="Name or label of the UI element (e.g. 'Submit Button')")
    description: str = Field(..., description="Description of the element's purpose or appearance")
    bounding_box: Optional[list[int]] = Field(
        default=None,
        description="Bounding box of the element as [x, y, width, height]"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score if available"
    )
    semantic_type: Optional[str] = Field(
        default=None,
        description="Semantic type of the UI element (e.g. 'button', 'text_box', 'menu', 'window', 'icon')"
    )


class ScreenAnalysis(BaseModel):
    """Structured analysis representation of the screen."""

    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the capture")
    width: int = Field(..., description="Width of the screen image in pixels")
    height: int = Field(..., description="Height of the screen image in pixels")
    ocr_text: Optional[str] = Field(default="", description="Aggregated text from OCR detection")
    ocr_regions: list[OCRTextRegion] = Field(
        default_factory=list,
        description="Detailed OCR text regions"
    )
    description: str = Field(..., description="General visual description of the screen")
    detected_elements: list[UIElement] = Field(
        default_factory=list,
        description="Visual UI elements detected"
    )
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Suggested next actions for the user/system"
    )


class VisionObservation(BaseModel):
    """State observation of the screen, returned to high-level planners."""

    analysis: ScreenAnalysis = Field(..., description="Detailed screen analysis contents")
    screenshot_path: Optional[str] = Field(default=None, description="Optional path where the screenshot is saved")
    state_hash: str = Field(..., description="MD5 hash representing the screen state")
    cache_hit: bool = Field(default=False, description="Whether the analysis result was served from cache")
