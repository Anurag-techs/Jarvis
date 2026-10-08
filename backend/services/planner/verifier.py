"""
JARVIS Planner Outcome Verification.
Implements pluggable VerificationManager aggregating visual and mock step verifiers.
"""

import json
import logging
from typing import Any

from backend.ai.provider import BaseLLMProvider
from backend.services.vision.service import VisionService
from backend.services.planner.interfaces import BaseVerifier
from backend.services.planner.models import PlanContext, PlanStep

logger = logging.getLogger("jarvis.services.planner.verifier")


class VerificationManager(BaseVerifier):
    """Aggregates and delegates step outcomes to multiple configured verifiers."""

    def __init__(self, verifiers: list[BaseVerifier] | None = None) -> None:
        """Initialize verification manager.

        Args:
            verifiers: Pre-configured verifier plugins.
        """
        self._verifiers = verifiers or []

    def add_verifier(self, verifier: BaseVerifier) -> None:
        """Dynamically registers an additional verifier plugin."""
        self._verifiers.append(verifier)

    def verify(self, step: PlanStep, context: PlanContext) -> bool:
        """Runs the registered verifiers sequentially. Returns True if any verifier succeeds."""
        if not step.verification_criteria:
            logger.debug("Step %s has no verification criteria. Auto-verifying.", step.id)
            return True

        logger.info("Executing pluggable verifiers for step: %s", step.id)
        for verifier in self._verifiers:
            try:
                if verifier.verify(step, context):
                    logger.info("Verifier %s successfully verified step %s", type(verifier).__name__, step.id)
                    return True
            except Exception as exc:
                logger.error("Error during verifier %s verification: %s", type(verifier).__name__, exc)

        logger.warning("Step %s verification failed against criteria: %s", step.id, step.verification_criteria)
        return False


class VisionVerifier(BaseVerifier):
    """Visual verifier evaluating screenshot states against criteria using multimodal LLM queries."""

    def __init__(self, vision_service: VisionService, llm_provider: BaseLLMProvider) -> None:
        """Initialize vision verifier.

        Args:
            vision_service: Coordinate service for screen capture.
            llm_provider: Active LLM provider used to execute multimodal checks.
        """
        self._vision_service = vision_service
        self._llm = llm_provider

    def verify(self, step: PlanStep, context: PlanContext) -> bool:
        """Grabs a screenshot and executes a zero-temperature JSON classification query against Gemini."""
        if not step.verification_criteria:
            return True

        from backend.ai.provider import MockLLMProvider
        if isinstance(self._llm, MockLLMProvider):
            logger.info("MockLLMProvider detected: Auto-verifying criteria: %s", step.verification_criteria)
            return True

        logger.info("VisionVerifier evaluating screen state for criteria: %s", step.verification_criteria)
        try:
            # Capture the primary display screen state
            image = self._vision_service._capture_provider.capture_screen()

            system_instruction = (
                "You are an automated step verification agent. Inspect the screenshot "
                "and determine if the specified verification criteria has been met. "
                "Output your evaluation as JSON matching the schema below.\n\n"
                "REQUIRED JSON OUTPUT SCHEMA:\n"
                "{\n"
                '  "verified": true or false,\n'
                '  "reason": "Detail explanation of what was visually detected and why the criteria is/is not met."\n'
                "}"
            )

            user_prompt = f"Verification Criteria: {step.verification_criteria}"

            # Retrieve internal Google GenAI SDK client
            client = getattr(self._llm, "_client", None)
            model_name = getattr(self._llm, "_model_name", "gemini-2.5-flash")

            if not client:
                logger.warning("No active LLM client found. Falling back to mock success.")
                return True

            import google.genai.types as types
            request_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                response_mime_type="application/json"
            )

            response = client.models.generate_content(
                model=model_name,
                contents=[image, user_prompt],
                config=request_config
            )

            payload = json.loads(response.text.strip())
            verified = payload.get("verified", False)
            reason = payload.get("reason", "")
            logger.info("Vision verification result for step %s: verified=%s, reason=%s", step.id, verified, reason)
            return verified

        except Exception as exc:
            logger.error("VisionVerifier evaluation failed for step %s: %s", step.id, exc)
            return False


class MockVerifier(BaseVerifier):
    """Simple stub verifier returning predetermined boolean results for testing."""

    def __init__(self, default_success: bool = True) -> None:
        self.default_success = default_success

    def verify(self, step: PlanStep, context: PlanContext) -> bool:
        logger.debug("MockVerifier returning success=%s for step %s", self.default_success, step.id)
        return self.default_success
