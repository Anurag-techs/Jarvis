"""
Coding Agent - Problem Parser.
Extracts structured ProgrammingProblem or MCQQuestion from screen captures using LLM analysis.
"""

import json
import logging
from PIL import Image
from typing import Tuple, Optional

from backend.ai.provider import BaseLLMProvider
from backend.agents.coding_agent.interfaces import BaseProblemParser
from backend.agents.coding_agent.models import (
    CodingPlatform,
    Example,
    MCQOption,
    MCQQuestion,
    ProgrammingLanguage,
    ProgrammingProblem,
)

logger = logging.getLogger("jarvis.agents.coding_agent.parser")

# Platform detection keywords mapped to enum values
_PLATFORM_KEYWORDS = {
    "leetcode": CodingPlatform.LEETCODE,
    "hackerrank": CodingPlatform.HACKERRANK,
    "codeforces": CodingPlatform.CODEFORCES,
    "geeksforgeeks": CodingPlatform.GEEKSFORGEEKS,
    "gfg": CodingPlatform.GEEKSFORGEEKS,
    "visual studio code": CodingPlatform.VSCODE,
    "vscode": CodingPlatform.VSCODE,
}

# Language detection keywords
_LANGUAGE_KEYWORDS = {
    "python": ProgrammingLanguage.PYTHON,
    "javascript": ProgrammingLanguage.JAVASCRIPT,
    "java": ProgrammingLanguage.JAVA,
    "c++": ProgrammingLanguage.CPP,
    "cpp": ProgrammingLanguage.CPP,
    "c#": ProgrammingLanguage.CSHARP,
    "csharp": ProgrammingLanguage.CSHARP,
    "golang": ProgrammingLanguage.GO,
    "go": ProgrammingLanguage.GO,
    "rust": ProgrammingLanguage.RUST,
    "typescript": ProgrammingLanguage.TYPESCRIPT,
}

_PARSE_SYSTEM_PROMPT = """You are a screen analysis engine. The user has captured a screenshot of a programming problem or a multiple-choice question.

Analyze the screen content and respond with a JSON object.

First, determine the content type:
- If it is a CODING PROBLEM, return:
{
  "type": "coding",
  "title": "Problem Title",
  "statement": "Full problem statement text",
  "constraints": ["constraint 1", "constraint 2"],
  "examples": [{"input": "...", "output": "...", "explanation": "..."}],
  "language": "python",
  "difficulty": "Easy/Medium/Hard",
  "function_signature": "def twoSum(self, nums: List[int], target: int) -> List[int]:"
}

- If it is a MULTIPLE-CHOICE QUESTION, return:
{
  "type": "mcq",
  "question_text": "What is the time complexity of binary search?",
  "options": [
    {"label": "A", "text": "O(n)"},
    {"label": "B", "text": "O(log n)"},
    {"label": "C", "text": "O(n^2)"},
    {"label": "D", "text": "O(1)"}
  ],
  "topic": "Data Structures and Algorithms"
}

Return ONLY the JSON object. No markdown, no explanation."""


class CodingProblemParser(BaseProblemParser):
    """Extracts problem or MCQ data from screen using LLM-powered analysis."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm = llm_provider

    def parse_screen(
        self,
        image: Image.Image,
        ocr_text: str,
        description: str,
    ) -> Tuple[Optional[ProgrammingProblem], Optional[MCQQuestion]]:
        logger.info("Parsing screen content for coding problem or MCQ...")

        # Detect platform from OCR text
        platform = self._detect_platform(ocr_text, description)
        logger.info("Detected platform: %s", platform)

        # Build user prompt with all available context
        user_prompt = (
            f"SCREEN DESCRIPTION:\n{description}\n\n"
            f"OCR TEXT:\n{ocr_text}\n\n"
            "Extract the programming problem or MCQ from this screen."
        )

        try:
            raw_response = self._llm.generate_raw_completion(
                user_prompt=user_prompt,
                system_prompt=_PARSE_SYSTEM_PROMPT,
                response_mime_type="application/json",
            )

            # Strip markdown fences if present
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            payload = json.loads(cleaned)
            content_type = payload.get("type", "coding")

            if content_type == "mcq":
                return None, self._build_mcq(payload, platform)
            else:
                return self._build_problem(payload, platform), None

        except Exception as e:
            logger.error("Failed to parse screen content: %s", e)
            # Attempt heuristic fallback from raw OCR
            return self._heuristic_parse(ocr_text, platform), None

    def _detect_platform(self, ocr_text: str, description: str) -> CodingPlatform:
        """Detects coding platform from OCR text and screen description."""
        combined = (ocr_text + " " + description).lower()
        for keyword, platform in _PLATFORM_KEYWORDS.items():
            if keyword in combined:
                return platform
        return CodingPlatform.GENERIC

    def _detect_language(self, text: str) -> ProgrammingLanguage:
        """Detects programming language from visible text."""
        lower = text.lower()
        for keyword, lang in _LANGUAGE_KEYWORDS.items():
            if keyword in lower:
                return lang
        return ProgrammingLanguage.PYTHON

    def _build_problem(self, payload: dict, platform: CodingPlatform) -> ProgrammingProblem:
        """Constructs a ProgrammingProblem from parsed JSON."""
        examples = []
        for ex in payload.get("examples", []):
            examples.append(Example(
                input=ex.get("input", ""),
                output=ex.get("output", ""),
                explanation=ex.get("explanation"),
            ))

        lang_str = payload.get("language", "python")
        language = self._detect_language(lang_str)

        return ProgrammingProblem(
            title=payload.get("title", "Unknown Problem"),
            statement=payload.get("statement", ""),
            constraints=payload.get("constraints", []),
            examples=examples,
            language=language,
            platform=platform,
            difficulty=payload.get("difficulty"),
            function_signature=payload.get("function_signature"),
        )

    def _build_mcq(self, payload: dict, platform: CodingPlatform) -> MCQQuestion:
        """Constructs an MCQQuestion from parsed JSON."""
        options = []
        for opt in payload.get("options", []):
            options.append(MCQOption(
                label=opt.get("label", ""),
                text=opt.get("text", ""),
            ))

        return MCQQuestion(
            question_text=payload.get("question_text", ""),
            options=options,
            topic=payload.get("topic"),
            platform=platform,
        )

    def _heuristic_parse(self, ocr_text: str, platform: CodingPlatform) -> ProgrammingProblem:
        """Fallback parser using raw OCR text when LLM parsing fails."""
        logger.warning("Using heuristic fallback parser from raw OCR text.")
        lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        title = lines[0] if lines else "Unknown Problem"
        statement = "\n".join(lines[1:]) if len(lines) > 1 else ocr_text

        return ProgrammingProblem(
            title=title,
            statement=statement,
            language=self._detect_language(ocr_text),
            platform=platform,
        )


# Backward compatibility alias
VisionProblemParser = CodingProblemParser
