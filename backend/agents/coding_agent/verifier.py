"""
Coding Agent - Verifier.
Verifies that generated code was successfully inserted into the editor
by checking for key code tokens in the post-insertion screen state.
"""

import logging
import re
from PIL import Image
from typing import List, Tuple

from backend.agents.coding_agent.interfaces import BaseCodingVerifier

logger = logging.getLogger("jarvis.agents.coding_agent.verifier")


class CodingVerifier(BaseCodingVerifier):
    """Confirms code insertion by comparing key tokens against post-insertion OCR text."""

    def verify_insertion(
        self,
        expected_code: str,
        image: Image.Image,
        ocr_text: str,
    ) -> Tuple[bool, str]:
        """Checks if key tokens from expected_code appear in the post-insertion OCR text.

        Verification strategy:
            1. Extract key tokens from the expected code (function names, class names,
               first significant line, keywords).
            2. Search for these tokens in the OCR text from the current screen.
            3. If enough tokens match (>= 50%), consider insertion verified.
        """
        logger.info("Verifying code insertion against screen OCR text...")

        if not ocr_text.strip():
            return False, "OCR text is empty — unable to verify insertion"

        key_tokens = self._extract_key_tokens(expected_code)
        if not key_tokens:
            logger.warning("No key tokens extracted from expected code. Assuming verification passed.")
            return True, "No key tokens to verify — assuming success"

        matches = 0
        missing = []
        for token in key_tokens:
            if token.lower() in ocr_text.lower():
                matches += 1
            else:
                missing.append(token)

        match_ratio = matches / len(key_tokens)
        logger.info(
            "Verification: %d/%d tokens matched (%.0f%%). Missing: %s",
            matches, len(key_tokens), match_ratio * 100, missing[:5],
        )

        if match_ratio >= 0.5:
            return True, f"Code verified: {matches}/{len(key_tokens)} key tokens found on screen"
        else:
            return False, f"Verification failed: only {matches}/{len(key_tokens)} tokens found. Missing: {missing[:3]}"

    def _extract_key_tokens(self, code: str) -> List[str]:
        """Extracts significant tokens from source code for verification matching."""
        tokens = []

        # Extract function names
        for match in re.finditer(r'def\s+(\w+)', code):
            tokens.append(match.group(1))

        # Extract class names
        for match in re.finditer(r'class\s+(\w+)', code):
            tokens.append(match.group(1))

        # Extract import names
        for match in re.finditer(r'import\s+(\w+)', code):
            tokens.append(match.group(1))

        # Extract the first non-empty, non-comment line as a signature token
        for line in code.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
                # Take first 3 words as a fingerprint
                words = stripped.split()[:3]
                token = " ".join(words)
                if len(token) >= 3:
                    tokens.append(token)
                break

        # Extract variable assignments (first significant variable name)
        for match in re.finditer(r'^(\w+)\s*=', code, re.MULTILINE):
            name = match.group(1)
            if name not in ("self", "_") and len(name) > 1:
                tokens.append(name)
                break  # Just the first one

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in tokens:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)

        logger.debug("Extracted %d key tokens: %s", len(unique), unique)
        return unique


# Backward compatibility alias
OCRCodingVerifier = CodingVerifier
