"""
Unit and Integration Tests for JARVIS Vision v1.0 framework.
"""

from datetime import datetime
import pytest
from PIL import Image

from backend.config.settings import AIConfig
from backend.core.exceptions import ProviderError
from backend.services.vision import (
    MockScreenCaptureProvider,
    MockOCRProvider,
    MockVisionProvider,
    LLMVisionProvider,
    VisionService,
    OCRTextRegion,
    UIElement,
    VisionObservation,
)
from backend.tools.vision_tool import VisionTool


def test_mock_screen_capture_provider():
    """Verify MockScreenCaptureProvider contract and region dimensions."""
    provider = MockScreenCaptureProvider()

    # Test full-screen default dimensions
    image_full = provider.capture_screen()
    assert isinstance(image_full, Image.Image)
    assert image_full.size == (1920, 1080)

    # Test region-based dimensions
    region = (100, 150, 400, 300)
    image_region = provider.capture_screen(region=region)
    assert isinstance(image_region, Image.Image)
    assert image_region.size == (400, 300)


def test_mock_ocr_provider():
    """Verify MockOCRProvider returns correct mock text blocks."""
    provider = MockOCRProvider()
    dummy_img = Image.new("RGB", (100, 100))

    regions = provider.detect_text(dummy_img)
    assert len(regions) > 0
    for r in regions:
        assert isinstance(r, OCRTextRegion)
        assert len(r.text) > 0
        assert r.confidence > 0.0
        assert len(r.bounding_box) == 4


def test_mock_vision_provider():
    """Verify MockVisionProvider returns expected mock descriptions and suggestions."""
    provider = MockVisionProvider()
    dummy_img = Image.new("RGB", (100, 100))

    description, elements, actions = provider.analyze_screen(dummy_img, "Welcome")
    assert "simulated" in description.lower() or "workspace" in description.lower()
    assert len(elements) > 0
    assert isinstance(elements[0], UIElement)
    assert len(actions) > 0


def test_llm_vision_provider_missing_key():
    """Verify LLMVisionProvider raises ProviderError if API key is missing."""
    config = AIConfig(provider="gemini", model_name="gemini-2.5-flash", api_key=None)
    with pytest.raises(ProviderError) as exc_info:
        LLMVisionProvider(config=config)
    assert "API key is unconfigured" in str(exc_info.value)


def test_vision_service_full_pipeline():
    """Verify VisionService integrates providers and returns structured VisionObservation."""
    capture = MockScreenCaptureProvider()
    ocr = MockOCRProvider()
    vision = MockVisionProvider()

    service = VisionService(
        capture_provider=capture,
        ocr_provider=ocr,
        vision_provider=vision
    )

    observation = service.analyze_current_screen(skip_ocr=False)

    assert isinstance(observation, VisionObservation)
    assert not observation.cache_hit
    assert observation.analysis.width == 1920
    assert observation.analysis.height == 1080
    assert "Welcome to JARVIS" in observation.analysis.ocr_text
    assert len(observation.analysis.detected_elements) == 3
    assert len(observation.analysis.suggested_actions) == 2


def test_vision_service_caching():
    """Verify that identical screens trigger a cache hit, and clear_cache() resets it."""
    capture = MockScreenCaptureProvider()
    ocr = MockOCRProvider()
    vision = MockVisionProvider()

    service = VisionService(
        capture_provider=capture,
        ocr_provider=ocr,
        vision_provider=vision
    )

    # First run (Cache Miss)
    obs1 = service.analyze_current_screen()
    assert not obs1.cache_hit

    # Second run (Cache Hit)
    obs2 = service.analyze_current_screen()
    assert obs2.cache_hit
    assert obs1.state_hash == obs2.state_hash

    # Third run after clearing cache (Cache Miss)
    service.clear_cache()
    obs3 = service.analyze_current_screen()
    assert not obs3.cache_hit


def test_vision_service_optional_ocr():
    """Verify that skip_ocr=True bypasses the OCR step and behaves correctly in cache."""
    capture = MockScreenCaptureProvider()
    ocr = MockOCRProvider()
    vision = MockVisionProvider()

    service = VisionService(
        capture_provider=capture,
        ocr_provider=ocr,
        vision_provider=vision
    )

    # Run with skip_ocr=True
    obs_no_ocr = service.analyze_current_screen(skip_ocr=True)
    assert not obs_no_ocr.cache_hit
    assert obs_no_ocr.analysis.ocr_text == ""
    assert len(obs_no_ocr.analysis.ocr_regions) == 0

    # Run with skip_ocr=False (since key is different (hash, ocr_performed), it should be a cache miss)
    obs_with_ocr = service.analyze_current_screen(skip_ocr=False)
    assert not obs_with_ocr.cache_hit
    assert "Welcome to JARVIS" in obs_with_ocr.analysis.ocr_text


def test_vision_tool_execution():
    """Verify VisionTool multi-action execution with mock services."""
    capture = MockScreenCaptureProvider()
    ocr = MockOCRProvider()
    vision = MockVisionProvider()
    service = VisionService(capture, ocr, vision)

    tool = VisionTool(vision_service=service)

    # Test 'capture' action
    res_cap = tool.execute(action="capture")
    assert res_cap.success
    assert "screenshots/screenshot_" in res_cap.message
    assert res_cap.data["width"] == 1920

    # Test 'ocr' action
    res_ocr = tool.execute(action="ocr")
    assert res_ocr.success
    assert "Welcome to JARVIS" in res_ocr.data["ocr_text"]

    # Test 'analyze' action
    res_an = tool.execute(action="analyze", skip_ocr=False)
    assert res_an.success
    assert "Visual analysis complete" in res_an.message
    assert res_an.data["analysis"]["width"] == 1920
    assert "online" in res_an.data["analysis"]["ocr_text"].lower()
