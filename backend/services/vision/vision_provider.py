"""
JARVIS Vision Model Providers.
Implements visual analysis of screens using LLMVisionProvider and Mock fallbacks.
"""

import json
import logging
from PIL import Image

from backend.config.settings import AIConfig
from backend.core.exceptions import ProviderError
from backend.services.vision.interfaces import BaseVisionProvider
from backend.services.vision.models import UIElement

logger = logging.getLogger("jarvis.services.vision.provider")


class LLMVisionProvider(BaseVisionProvider):
    """Vision Provider utilizing Google Gemini multimodal models."""

    def __init__(self, config: AIConfig) -> None:
        """Initialize the LLMVisionProvider.

        Args:
            config: Encapsulated AI provider configuration.
        """
        self._config = config
        self._model_name = config.model_name
        self._client = None

        if not config.api_key:
            raise ProviderError(
                message="Google Gemini API key is unconfigured. Set GEMINI_API_KEY in .env file.",
                details={"provider": "gemini", "model": self._model_name},
            )

    def _get_client(self):
        """Lazy-initializes and caches the Google GenAI SDK client."""
        if self._client is None:
            try:
                import google.genai as genai
                self._client = genai.Client(api_key=self._config.api_key)
            except ImportError as exc:
                raise ProviderError(
                    message=(
                        "google-genai SDK is not installed. "
                        "Run: pip install google-genai>=1.0.0"
                    ),
                    details={"provider": "gemini", "model": self._model_name},
                ) from exc
            except Exception as exc:
                raise ProviderError(
                    message=f"Failed to initialize Google GenAI SDK client: {exc}",
                    details={"provider": "gemini", "model": self._model_name},
                ) from exc
        return self._client

    def analyze_screen(
        self,
        image: Image.Image,
        ocr_text: str | None = None
    ) -> tuple[str, list[UIElement], list[str]]:
        """Sends the screenshot image and helper OCR text to Gemini for structured analysis.

        Args:
            image: PIL Image of the screen.
            ocr_text: Optional OCR text helper.

        Returns:
            Tuple of (description, detected_elements, suggested_actions).
        """
        client = self._get_client()

        try:
            import google.genai.types as types

            system_instruction = (
                "You are an expert visual AI assistant analyzing a computer screen screenshot.\n"
                "You must analyze the screenshot image and the provided OCR text (if present), identify key visual UI elements, "
                "and output a structured JSON response matching the schema below.\n\n"
                "REQUIRED JSON OUTPUT SCHEMA:\n"
                "{\n"
                '  "description": "General visual description of what is on the screen.",\n'
                '  "detected_elements": [\n'
                '    {\n'
                '      "label": "Name of UI element (e.g. Submit Button, Text Box)",\n'
                '      "description": "Description of the element and its location.",\n'
                '      "bounding_box": [x, y, width, height],\n'
                '      "confidence": 0.95,\n'
                '      "semantic_type": "type of UI element (e.g. button, text_box, menu, window, icon)"\n'
                '    }\n'
                '  ],\n'
                '  "suggested_actions": [\n'
                '    "Action 1",\n'
                '    "Action 2"\n'
                '  ]\n'
                "}"
            )

            ocr_info = f"\nOCR Detected Text in the image:\n{ocr_text}\n" if ocr_text else "\nNo OCR text available."
            user_prompt = f"Please analyze this screen image.{ocr_info}"

            request_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=self._config.temperature,
                response_mime_type="application/json",
            )

            logger.info("Sending screen image to Gemini vision model: %s", self._model_name)
            response = client.models.generate_content(
                model=self._model_name,
                contents=[image, user_prompt],
                config=request_config,
            )

            if not response or not response.text:
                raise ProviderError(
                    message="LLMVisionProvider returned empty content.",
                    details={"model": self._model_name}
                )

            payload = json.loads(response.text.strip())

            description = payload.get("description", "No description provided.")
            raw_elements = payload.get("detected_elements", [])
            elements = []
            for item in raw_elements:
                elements.append(
                    UIElement(
                        label=item.get("label", ""),
                        description=item.get("description", ""),
                        bounding_box=item.get("bounding_box"),
                        confidence=item.get("confidence"),
                        semantic_type=item.get("semantic_type")
                    )
                )
            suggested_actions = payload.get("suggested_actions", [])

            logger.info("Successfully analyzed screen visual layout.")
            return description, elements, suggested_actions

        except Exception as exc:
            logger.error("LLMVisionProvider screen analysis failed: %s", exc)
            raise ProviderError(
                message=f"Vision model analysis failed: {exc}",
                details={"provider": "gemini", "model": self._model_name}
            ) from exc


class MockVisionProvider(BaseVisionProvider):
    """Mock vision provider returning pre-determined responses for testing."""

    def analyze_screen(
        self,
        image: Image.Image,
        ocr_text: str | None = None
    ) -> tuple[str, list[UIElement], list[str]]:
        """Returns dummy analysis payload."""
        logger.debug("MockVisionProvider returning mock screen analysis.")
        description = "A simulated workspace showing active system applications."
        elements = [
            UIElement(
                label="Application Header",
                description="Visual header area of the primary window",
                bounding_box=[50, 50, 200, 30],
                confidence=0.99,
                semantic_type="window"
            ),
            UIElement(
                label="Status Message",
                description="Status text showing online state",
                bounding_box=[50, 100, 130, 20],
                confidence=0.95,
                semantic_type="text_box"
            ),
            UIElement(
                label="Action Area",
                description="Clickable action trigger button",
                bounding_box=[400, 300, 150, 30],
                confidence=0.92,
                semantic_type="button"
            )
        ]
        actions = [
            "Trigger Action Area",
            "Read Status Message details"
        ]
        return description, elements, actions
